# Start of deoldify/filters.py
from fastai.basic_data import DatasetType
from fastai.basic_train import Learner
from abc import ABC, abstractmethod
from fastai.core import *
from fastai.vision import * # This should cover most vision related imports
# from fastai.vision.image import * # Covered by above
# from fastai.vision.data import * # Covered by above
# from fastai import * # Covered by above
import cv2
from PIL import Image as PilImage # Standard alias
from deoldify import device as device_settings # Specific to deoldify
import logging
import torch # Explicit import for torch.device, torch.cat etc.
from torch import Tensor # Explicit import
import torch.nn as nn # Explicit import
from typing import List, Tuple # Explicit import for type hints

# Class definitions for IFilter, BaseFilter, ColorizerFilter, MasterFilter
# (These should be copied verbatim from the existing file, assuming they are correct and not duplicated)

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

    def _transform(self, image: PilImage) -> PilImage: return image
    def _scale_to_square(self, orig: PilImage, targ: int) -> PilImage:
        targ_sz = (targ, targ)
        return orig.resize(targ_sz, resample=PilImage.BILINEAR)
    def _get_model_ready_image(self, orig: PilImage, sz: int) -> PilImage:
        result = self._scale_to_square(orig, sz)
        result = self._transform(result)
        return result
    def _model_process(self, orig: PilImage, sz: int) -> PilImage:
        model_image = self._get_model_ready_image(orig, sz)
        x = pil2tensor(model_image, np.float32)
        x = x.to(self.device); x.div_(255)
        x, y = self.norm((x, x), do_x=True)
        try:
            result = self.learn.pred_batch(ds_type=DatasetType.Valid, batch=(x[None], y[None]), reconstruct=True)
        except RuntimeError as rerr:
            if 'memory' not in str(rerr): raise rerr
            logging.warn('Warning: render_factor was set too high, and out of memory error resulted. Returning original image.')
            return model_image
        out = result[0]; out = self.denorm(out.px, do_x=False)
        out = image2np(out * 255).astype(np.uint8)
        return PilImage.fromarray(out)
    def _unsquare(self, image: PilImage, orig: PilImage) -> PilImage:
        targ_sz = orig.size
        image = image.resize(targ_sz, resample=PilImage.BILINEAR)
        return image

class ColorizerFilter(BaseFilter):
    def __init__(self, learn: Learner, stats: tuple = imagenet_stats):
        super().__init__(learn=learn, stats=stats)
        self.render_base = 16
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        render_sz = render_factor * self.render_base
        model_image = self._model_process(orig=filtered_image, sz=render_sz)
        raw_color = self._unsquare(model_image, orig_image)
        if post_process: return self._post_process(raw_color, orig_image)
        else: return raw_color
    def _transform(self, image: PilImage) -> PilImage: return image.convert('LA').convert('RGB')
    def _post_process(self, raw_color: PilImage, orig: PilImage) -> PilImage:
        color_np = np.asarray(raw_color); orig_np = np.asarray(orig)
        color_yuv = cv2.cvtColor(color_np, cv2.COLOR_RGB2YUV)
        orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2YUV)
        hires = np.copy(orig_yuv); hires[:, :, 1:3] = color_yuv[:, :, 1:3]
        final = cv2.cvtColor(hires, cv2.COLOR_YUV2RGB)
        return PilImage.fromarray(final)

class MasterFilter(BaseFilter): 
    def __init__(self, filters: List[IFilter], render_factor: int):
        self.filters = filters
        self.render_factor = render_factor
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        render_factor = self.render_factor if render_factor is None else render_factor
        for f in self.filters: 
            filtered_image = f.filter(orig_image, filtered_image, render_factor, post_process)
        return filtered_image

# Single import for denorm_lab_mean_std_tensor
from .augs import denorm_lab_mean_std_tensor

# Single definition of VideoColorizerFilter (use the last one from the file which is assumed to be the most correct)
class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, **kwargs):
        super().__init__(**kwargs) 
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu") 
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")

    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor, post_process)
    
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        return self.P(filtered_image, render_factor=render_factor, post_process=post_process)

    def process_sequence(self, frame_sequence: List[PilImage], render_factor: int = None, post_process: bool = True) -> PilImage:
        if len(frame_sequence) != self.n_frames_input:
            raise ValueError(f"Expected {self.n_frames_input} frames, got {len(frame_sequence)}")

        current_render_factor = render_factor if render_factor is not None else self.render_factor
        
        processed_frames_l_tensors = [] 
        middle_frame_idx = self.n_frames_input // 2
        orig_middle_frame_pil = frame_sequence[middle_frame_idx]
        
        target_w_render, target_h_render = orig_middle_frame_pil.size 
        if current_render_factor is not None:
            target_w_render, target_h_render = self.get_render_factor_target_size(orig_middle_frame_pil, current_render_factor)
        else: 
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 0 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 0 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)

        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = ToTensor()(frame_pil_l_resized) # Use ToTensor from fastai.vision
            processed_frames_l_tensors.append(frame_tensor_l)
        
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): 
            ab_output_tensor = self.model(input_tensor).squeeze(0) 

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        # Use FastAI's Image class for its convenient .resize().data method
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = ToTensor()(middle_frame_orig_l_pil) # Use ToTensor from fastai.vision
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = ToPILImage()(full_img_tensor.cpu()) # Use ToPILImage from fastai.vision
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        new_w = new_w - (new_w % 16) if new_w > 0 else 16
        new_h = new_h - (new_h % 16) if new_h > 0 else 16
        return max(16, new_w), max(16, new_h) 

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        t_l_scaled = t_l * 100.0 
        return torch.cat([t_l_scaled, t_ab], dim=0)

# End of deoldify/filters.py
