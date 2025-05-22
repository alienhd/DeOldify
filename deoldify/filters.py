from fastai.basic_data import DatasetType
from fastai.basic_train import Learner
from abc import ABC, abstractmethod
from fastai.core import *
from fastai.vision import *
from fastai.vision.image import *
from fastai.vision.data import *
from fastai import *
import cv2
from PIL import Image as PilImage
from deoldify import device as device_settings
import logging
from pathlib import Path
from typing import Optional, List, Tuple

class IFilter(ABC):
    @abstractmethod
    def filter(
        self, orig_image: PilImage, filtered_image: PilImage, render_factor: int
    ) -> PilImage:
        pass


class BaseFilter(IFilter):
    def __init__(self, learn: Learner, stats: tuple = imagenet_stats):
        super().__init__()
        self.learn = learn
        
        if not device_settings.is_gpu():
            self.learn.model = self.learn.model.cpu()
        
        self.device = next(self.learn.model.parameters()).device
        self.norm, self.denorm = normalize_funcs(*stats)

    def _transform(self, image: PilImage) -> PilImage:
        return image

    def _scale_to_square(self, orig: PilImage, targ: int) -> PilImage:
        targ_sz = (targ, targ)
        return orig.resize(targ_sz, resample=PIL.Image.BILINEAR)

    def _get_model_ready_image(self, orig: PilImage, sz: int) -> PilImage:
        result = self._scale_to_square(orig, sz)
        result = self._transform(result)
        return result

    def _model_process(self, orig: PilImage, sz: int) -> PilImage:
        model_image = self._get_model_ready_image(orig, sz)
        x = pil2tensor(model_image, np.float32)
        x = x.to(self.device)
        x.div_(255)
        x, y = self.norm((x, x), do_x=True)
        
        try:
            result = self.learn.pred_batch(
                ds_type=DatasetType.Valid, batch=(x[None], y[None]), reconstruct=True
            )
        except RuntimeError as rerr:
            if 'memory' not in str(rerr):
                raise rerr
            logging.warn('Warning: render_factor was set too high, and out of memory error resulted. Returning original image.')
            return model_image
            
        out = result[0]
        out = self.denorm(out.px, do_x=False)
        out = image2np(out * 255).astype(np.uint8)
        return PilImage.fromarray(out)

    def _unsquare(self, image: PilImage, orig: PilImage) -> PilImage:
        targ_sz = orig.size
        image = image.resize(targ_sz, resample=PIL.Image.BILINEAR)
        return image


class ColorizerFilter(BaseFilter):
    def __init__(self, learn: Learner, stats: tuple = imagenet_stats):
        super().__init__(learn=learn, stats=stats)
        self.render_base = 16

    def filter(
        self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        render_sz = render_factor * self.render_base
        model_image = self._model_process(orig=filtered_image, sz=render_sz)
        raw_color = self._unsquare(model_image, orig_image)

        if post_process:
            return self._post_process(raw_color, orig_image)
        else:
            return raw_color

    def _transform(self, image: PilImage) -> PilImage:
        return image.convert('LA').convert('RGB')

    def _post_process(self, raw_color: PilImage, orig: PilImage) -> PilImage:
        color_np = np.asarray(raw_color)
        orig_np = np.asarray(orig)
        color_yuv = cv2.cvtColor(color_np, cv2.COLOR_RGB2YUV)
        orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2YUV)
        hires = np.copy(orig_yuv)
        hires[:, :, 1:3] = color_yuv[:, :, 1:3]
        final = cv2.cvtColor(hires, cv2.COLOR_YUV2RGB)
        final = PilImage.fromarray(final)
        return final


class MasterFilter(BaseFilter):
    def __init__(self, filters: List[IFilter], render_factor: int):
        self.filters = filters
        self.render_factor = render_factor

    def filter(
        self, orig_image: PilImage, filtered_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        render_factor = self.render_factor if render_factor is None else render_factor
        for filter_item in self.filters: 
            filtered_image = filter_item.filter(orig_image, filtered_image, render_factor, post_process)
        return filtered_image

# Imports for VideoColorizerFilter
import torch
from torch import Tensor 
import torch.nn as nn 
# PilImage is already imported as from PIL import Image as PilImage
# IFilter is defined in this file

class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, stats: tuple = imagenet_stats, 
                 debug_dir: Optional[str] = None, debug_frame_prefix: Optional[str] = 'frame', **kwargs):
        super().__init__(**kwargs) 
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        self.debug_frame_prefix = debug_frame_prefix
        self.debug_dir = Path(debug_dir) if debug_dir else None
        
        if self.debug_dir:
            try:
                self.debug_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logging.error(f"Could not create debug directory {self.debug_dir}. Error: {e}")
                self.debug_dir = None 
            
        try:
            self.device = next(model.parameters()).device
        except StopIteration: 
            self.device = torch.device("cpu") 
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")
        self.norm, self.denorm = normalize_funcs(*stats)

    def _save_debug_tensor_as_image(self, tensor_data: Tensor, filename_suffix: str, channel_names: Optional[List[str]] = None):
        if not self.debug_dir:
            return

        save_path_base = self.debug_dir / f"{self.debug_frame_prefix}_{filename_suffix}"
        
        try:
            tensor_to_save = tensor_data.clone().cpu().detach()

            if tensor_to_save.ndim == 2: 
                tensor_to_save = tensor_to_save.unsqueeze(0)
            
            if tensor_to_save.ndim == 3 and tensor_to_save.shape[0] == 1: 
                min_val, max_val = tensor_to_save.min(), tensor_to_save.max()
                # Normalize only if not already in a typical image range (e.g. 0-1 or 0-255)
                # and also ensure max_val is not equal to min_val to prevent division by zero.
                if not ( (torch.isclose(min_val, torch.tensor(0.0), atol=1e-3) and torch.isclose(max_val, torch.tensor(1.0), atol=1e-3)) or \
                           (torch.isclose(min_val, torch.tensor(0.0), atol=1e-1) and torch.isclose(max_val, torch.tensor(255.0), atol=1e-1)) ): # Approx check for 0-1 or 0-255
                    if torch.abs(max_val - min_val) < 1e-6 : # Avoid division by zero if tensor is constant
                        vis_tensor = tensor_to_save - min_val # Results in all zeros
                    else:
                        vis_tensor = (tensor_to_save - min_val) / (max_val - min_val + 1e-6) # Added epsilon for safety
                else: # Already in 0-1 or 0-255 range
                    vis_tensor = tensor_to_save / 255.0 if max_val > 1.0 else tensor_to_save # Normalize 0-255 to 0-1
                
                img = vision.transform.ToPILImage()(vis_tensor.clamp(0.0, 1.0)) # Clamp to ensure valid range for ToPILImage
                img.save(str(save_path_base) + ".png")

            elif tensor_to_save.ndim == 3: 
                if channel_names and len(channel_names) == tensor_to_save.shape[0]:
                    for i in range(tensor_to_save.shape[0]):
                        channel_tensor_single = tensor_to_save[i].unsqueeze(0) 
                        min_val, max_val = channel_tensor_single.min(), channel_tensor_single.max()
                        if not ( (torch.isclose(min_val, torch.tensor(0.0), atol=1e-3) and torch.isclose(max_val, torch.tensor(1.0), atol=1e-3)) or \
                                   (torch.isclose(min_val, torch.tensor(0.0), atol=1e-1) and torch.isclose(max_val, torch.tensor(255.0), atol=1e-1)) ):
                             if torch.abs(max_val - min_val) < 1e-6 :
                                 vis_tensor = channel_tensor_single - min_val
                             else:
                                 vis_tensor = (channel_tensor_single - min_val) / (max_val - min_val + 1e-6) # Added epsilon
                        else:
                            vis_tensor = channel_tensor_single / 255.0 if max_val > 1.0 else channel_tensor_single
                        img = vision.transform.ToPILImage()(vis_tensor.clamp(0.0, 1.0))
                        img.save(str(save_path_base) + f"_{channel_names[i]}.png")
                elif tensor_to_save.shape[0] == 3: 
                    if filename_suffix == "merged_LAB_prenorm":
                        # L channel (0-100) scaled to 0-1 for saving
                        img_l_tensor = (tensor_to_save[0]/100.0).clamp(0.0, 1.0).unsqueeze(0)
                        img_l = vision.transform.ToPILImage()(img_l_tensor)
                        img_l.save(str(save_path_base) + "_L.png")
                        # A and B channels normalized individually for visualization
                        for i, chan_name in enumerate(['A', 'B']):
                            channel_tensor_single = tensor_to_save[i+1]
                            min_val_c, max_val_c = channel_tensor_single.min(), channel_tensor_single.max()
                            if torch.abs(max_val_c - min_val_c) < 1e-6:
                                vis_tensor = channel_tensor_single - min_val_c
                            else:
                                vis_tensor = (channel_tensor_single - min_val_c) / (max_val_c - min_val_c + 1e-6) # Added epsilon
                            img_chan = vision.transform.ToPILImage()(vis_tensor.clamp(0.0,1.0).unsqueeze(0))
                            img_chan.save(str(save_path_base) + f"_{chan_name}_norm.png")
                    else: # Assumed denormalized RGB (e.g. merged_LAB_postnorm) or other 3-channel
                        img = vision.transform.ToPILImage()(tensor_to_save.clamp(0.0, 1.0)) 
                        img.save(str(save_path_base) + ".png")
                else: # Fallback for other multi-channel tensors
                    for i in range(tensor_to_save.shape[0]):
                        channel_tensor_single = tensor_to_save[i]
                        min_val, max_val = channel_tensor_single.min(), channel_tensor_single.max()
                        if not ( (torch.isclose(min_val, torch.tensor(0.0), atol=1e-3) and torch.isclose(max_val, torch.tensor(1.0), atol=1e-3)) or \
                                   (torch.isclose(min_val, torch.tensor(0.0), atol=1e-1) and torch.isclose(max_val, torch.tensor(255.0), atol=1e-1)) ):
                             if torch.abs(max_val - min_val) < 1e-6:
                                 vis_tensor = channel_tensor_single - min_val
                             else:
                                 vis_tensor = (channel_tensor_single - min_val) / (max_val - min_val + 1e-6) # Added epsilon
                        else:
                             vis_tensor = channel_tensor_single / 255.0 if max_val > 1.0 else channel_tensor_single
                        img = vision.transform.ToPILImage()(vis_tensor.clamp(0.0, 1.0).unsqueeze(0))
                        img.save(str(save_path_base) + f"_channel_{i}.png")
            else:
                logging.warning(f"Could not save debug image for {filename_suffix}. Tensor dimension not supported: {tensor_to_save.ndim} with shape {tensor_to_save.shape}")
        except Exception as e:
            logging.warning(f"Failed to save debug image {save_path_base} for {filename_suffix}. Error: {e}")

    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor, post_process)
    
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        return self.P(filtered_image, render_factor=render_factor, post_process=post_process)

    def process_sequence(self, frame_sequence: List[PilImage], render_factor: int = None, post_process: bool = True) -> PilImage:
        if len(frame_sequence) != self.n_frames_input:
            raise ValueError(f"Expected {self.n_frames_input} frames, got {len(frame_sequence)}")

        current_render_factor = render_factor if render_factor is not None else self.render_factor
        
        processed_frames_l = [] 
        middle_frame_idx = self.n_frames_input // 2
        orig_middle_frame_pil = frame_sequence[middle_frame_idx]
        
        if current_render_factor is not None:
            target_w_render, target_h_render = self.get_render_factor_target_size(orig_middle_frame_pil, current_render_factor)
        else:
            temp_w, temp_h = orig_middle_frame_pil.size
            target_w_render, target_h_render = temp_w - temp_w % 16, temp_h - temp_h % 16
            if target_w_render == 0 or target_h_render == 0: 
                target_w_render, target_h_render = orig_middle_frame_pil.size

        for idx, frame_pil in enumerate(frame_sequence):
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l.append(frame_tensor_l)
            if idx == middle_frame_idx and self.debug_dir: 
                 self._save_debug_tensor_as_image(frame_tensor_l, "input_L_middle_resized")
        
        input_tensor = torch.cat(processed_frames_l, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)
        
        if self.debug_dir and middle_frame_idx < len(processed_frames_l): 
             self._save_debug_tensor_as_image(processed_frames_l[middle_frame_idx], "input_tensor_middle_channel_L")
        
        with torch.no_cache():
            ab_output_tensor = self.model(input_tensor).squeeze(0) 
        if self.debug_dir: 
            self._save_debug_tensor_as_image(ab_output_tensor, "output_AB_raw", channel_names=['A', 'B'])
        
        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        color_img_ab = Image(ab_output_tensor.cpu().detach()).resize(orig_middle_frame_pil.size) 
        ab_output_resized = color_img_ab.data 
        if self.debug_dir: 
            self._save_debug_tensor_as_image(ab_output_resized, "output_AB_resized", channel_names=['A', 'B'])
            
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor_prenorm = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if self.debug_dir: 
             self._save_debug_tensor_as_image(full_img_tensor_prenorm.clone(), "merged_LAB_prenorm")

        if post_process:
            full_img_tensor = self.denorm(full_img_tensor_prenorm.clone()) 
            if self.debug_dir: 
                self._save_debug_tensor_as_image(full_img_tensor.clone(), "merged_LAB_postnorm")
        else:
            full_img_tensor = full_img_tensor_prenorm
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu().detach())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        new_w = new_w - new_w % 16
        new_h = new_h - new_h % 16
        return max(16, new_w), max(16, new_h) 

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        t_l_scaled = t_l * 100.0 
        return torch.cat([t_l_scaled, t_ab], dim=0)
