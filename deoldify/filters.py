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
        # a simple stretch to fit a square really makes a big difference in rendering quality/consistency.
        # I've tried padding to the square as well (reflect, symetric, constant, etc).  Not as good!
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

    # This takes advantage of the fact that human eyes are much less sensitive to
    # imperfections in chrominance compared to luminance.  This means we can
    # save a lot on memory and processing in the model, yet get a great high
    # resolution result at the end.  This is primarily intended just for
    # inference
    def _post_process(self, raw_color: PilImage, orig: PilImage) -> PilImage:
        color_np = np.asarray(raw_color)
        orig_np = np.asarray(orig)
        color_yuv = cv2.cvtColor(color_np, cv2.COLOR_RGB2YUV)
        # do a black and white transform first to get better luminance values
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
        for filter in self.filters:
            filtered_image = filter.filter(orig_image, filtered_image, render_factor, post_process)

        return filtered_image


# Imports for VideoColorizerFilter are mostly covered by existing imports.
# torch, Tensor, nn, Image (FastAI), vision_transform, List, Tuple, logging, PilImage are already available
# via existing `from fastai.vision import *` and `from fastai.core import *`
from .augs import denorm_lab_mean_std_tensor # Specific import needed
# IFilter is defined in this file. nn.Module is available from fastai.vision too.


class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, **kwargs):
        super().__init__(**kwargs) 
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        try:
            self.device = next(model.parameters()).device
        except StopIteration: # Model has no parameters
            self.device = torch.device("cpu") 
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")


    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        # This method processes a single image by duplicating it n_frames_input times
        # to form a sequence, then calling process_sequence.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor=render_factor, post_process=post_process)
    
    # Required by IFilter interface
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # We process 'filtered_image' as it's the output from the previous stage (or the input if this is the first filter).
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 16 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 16 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)

        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack L-channel tensors: each is (1, H_r, W_r). Concatenate along channel dim.
        # Resulting shape (n_frames_input, H_r, W_r), then add batch dim.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): 
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Use fastai Image for resizing tensor
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of 16 and at least 16
        new_w = max(16, new_w - (new_w % 16))
        new_h = max(16, new_h - (new_h % 16))
        return new_w, new_h

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        t_l_scaled = t_l * 100.0 # Scale L channel to [0, 100]
        return torch.cat([t_l_scaled, t_ab], dim=0) # Result (3, H, W)


# Imports for VideoColorizerFilter are mostly covered by existing imports.
# torch, Tensor, nn, Image (FastAI), vision_transform, List, Tuple, logging, PilImage are already available
# via existing `from fastai.vision import *` and `from fastai.core import *`
from .augs import denorm_lab_mean_std_tensor # Specific import needed
# IFilter is defined in this file. nn.Module is available from fastai.vision too.


class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, **kwargs):
        super().__init__(**kwargs) 
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        try:
            self.device = next(model.parameters()).device
        except StopIteration: # Model has no parameters
            self.device = torch.device("cpu") 
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")


    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        # This method processes a single image by duplicating it n_frames_input times
        # to form a sequence, then calling process_sequence.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor=render_factor, post_process=post_process)
    
    # Required by IFilter interface
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # We process 'filtered_image' as it's the output from the previous stage (or the input if this is the first filter).
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 16 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 16 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)

        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack L-channel tensors: each is (1, H_r, W_r). Concatenate along channel dim.
        # Resulting shape (n_frames_input, H_r, W_r), then add batch dim.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): 
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Use fastai Image for resizing tensor
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of 16 and at least 16
        new_w = max(16, new_w - (new_w % 16))
        new_h = max(16, new_h - (new_h % 16))
        return new_w, new_h

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        t_l_scaled = t_l * 100.0 # Scale L channel to [0, 100]
        return torch.cat([t_l_scaled, t_ab], dim=0) # Result (3, H, W)


# Imports for VideoColorizerFilter are mostly covered by existing imports.
# torch, Tensor, nn, Image (FastAI), vision_transform, List, Tuple, logging, PilImage are already available
# via existing `from fastai.vision import *` and `from fastai.core import *`
from .augs import denorm_lab_mean_std_tensor # Specific import needed
# IFilter is defined in this file. nn.Module is available from fastai.vision too.


class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, **kwargs):
        super().__init__(**kwargs) 
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        try:
            self.device = next(model.parameters()).device
        except StopIteration: # Model has no parameters
            self.device = torch.device("cpu") 
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")


    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        # This method processes a single image by duplicating it n_frames_input times
        # to form a sequence, then calling process_sequence.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor=render_factor, post_process=post_process)
    
    # Required by IFilter interface
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # We process 'filtered_image' as it's the output from the previous stage (or the input if this is the first filter).
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 16 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 16 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)

        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack L-channel tensors: each is (1, H_r, W_r). Concatenate along channel dim.
        # Resulting shape (n_frames_input, H_r, W_r), then add batch dim.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): 
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Use fastai Image for resizing tensor
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of 16 and at least 16
        new_w = max(16, new_w - (new_w % 16))
        new_h = max(16, new_h - (new_h % 16))
        return new_w, new_h

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        t_l_scaled = t_l * 100.0 # Scale L channel to [0, 100]
        return torch.cat([t_l_scaled, t_ab], dim=0) # Result (3, H, W)


# Imports for VideoColorizerFilter are mostly covered by existing imports.
# torch, Tensor, nn, Image (FastAI), vision_transform, List, Tuple, logging, PilImage are already available
# via existing `from fastai.vision import *` and `from fastai.core import *`
from .augs import denorm_lab_mean_std_tensor # Specific import needed
# IFilter is defined in this file. nn.Module is available from fastai.vision too.


class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, **kwargs):
        super().__init__(**kwargs) 
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        try:
            self.device = next(model.parameters()).device
        except StopIteration: # Model has no parameters
            self.device = torch.device("cpu") 
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")


    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        # This method processes a single image by duplicating it n_frames_input times
        # to form a sequence, then calling process_sequence.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor=render_factor, post_process=post_process)
    
    # Required by IFilter interface
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # We process 'filtered_image' as it's the output from the previous stage (or the input if this is the first filter).
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 16 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 16 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)

        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack L-channel tensors: each is (1, H_r, W_r). Concatenate along channel dim.
        # Resulting shape (n_frames_input, H_r, W_r), then add batch dim.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): 
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Use fastai Image for resizing tensor
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of 16 and at least 16
        new_w = max(16, new_w - (new_w % 16))
        new_h = max(16, new_h - (new_h % 16))
        return new_w, new_h

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        t_l_scaled = t_l * 100.0 # Scale L channel to [0, 100]
        return torch.cat([t_l_scaled, t_ab], dim=0) # Result (3, H, W)


# Imports for VideoColorizerFilter are mostly covered by existing imports.
# torch, Tensor, nn, Image (FastAI), vision_transform, List, Tuple, logging, PilImage are already available
# via existing `from fastai.vision import *` and `from fastai.core import *`
from .augs import denorm_lab_mean_std_tensor # Specific import needed
# IFilter is defined in this file. nn.Module is available from fastai.vision too.


class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, **kwargs):
        super().__init__(**kwargs) 
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        try:
            self.device = next(model.parameters()).device
        except StopIteration: # Model has no parameters
            self.device = torch.device("cpu") 
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")


    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        # This method processes a single image by duplicating it n_frames_input times
        # to form a sequence, then calling process_sequence.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor=render_factor, post_process=post_process)
    
    # Required by IFilter interface
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # We process 'filtered_image' as it's the output from the previous stage (or the input if this is the first filter).
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 16 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 16 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)

        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack L-channel tensors: each is (1, H_r, W_r). Concatenate along channel dim.
        # Resulting shape (n_frames_input, H_r, W_r), then add batch dim.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): 
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Use fastai Image for resizing tensor
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of 16 and at least 16
        new_w = max(16, new_w - (new_w % 16))
        new_h = max(16, new_h - (new_h % 16))
        return new_w, new_h

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        t_l_scaled = t_l * 100.0 # Scale L channel to [0, 100]
        return torch.cat([t_l_scaled, t_ab], dim=0) # Result (3, H, W)


# Imports for VideoColorizerFilter are mostly covered by existing imports.
# torch, Tensor, nn, Image (FastAI), vision_transform, List, Tuple, logging, PilImage are already available
# via existing `from fastai.vision import *` and `from fastai.core import *`
from .augs import denorm_lab_mean_std_tensor # Specific import needed
# IFilter is defined in this file. nn.Module is available from fastai.vision too.


class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, **kwargs):
        super().__init__(**kwargs) 
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        try:
            self.device = next(model.parameters()).device
        except StopIteration: # Model has no parameters
            self.device = torch.device("cpu") 
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")


    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        # This method processes a single image by duplicating it n_frames_input times
        # to form a sequence, then calling process_sequence.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor=render_factor, post_process=post_process)
    
    # Required by IFilter interface
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # We process 'filtered_image' as it's the output from the previous stage (or the input if this is the first filter).
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 16 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 16 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)

        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack L-channel tensors: each is (1, H_r, W_r). Concatenate along channel dim.
        # Resulting shape (n_frames_input, H_r, W_r), then add batch dim.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): 
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Use fastai Image for resizing tensor
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of 16 and at least 16
        new_w = max(16, new_w - (new_w % 16))
        new_h = max(16, new_h - (new_h % 16))
        return new_w, new_h

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        t_l_scaled = t_l * 100.0 # Scale L channel to [0, 100]
        return torch.cat([t_l_scaled, t_ab], dim=0) # Result (3, H, W)


# Imports for VideoColorizerFilter are mostly covered by existing imports.
# torch, Tensor, nn, Image (FastAI), vision_transform, List, Tuple, logging, PilImage are already available
# via existing `from fastai.vision import *` and `from fastai.core import *`
from .augs import denorm_lab_mean_std_tensor # Specific import needed
# IFilter is defined in this file. nn.Module is available from fastai.vision too.


class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, **kwargs):
        super().__init__(**kwargs) 
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        try:
            self.device = next(model.parameters()).device
        except StopIteration: # Model has no parameters
            self.device = torch.device("cpu") 
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")


    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        # This method processes a single image by duplicating it n_frames_input times
        # to form a sequence, then calling process_sequence.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor=render_factor, post_process=post_process)
    
    # Required by IFilter interface
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # We process 'filtered_image' as it's the output from the previous stage (or the input if this is the first filter).
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 16 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 16 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)

        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack L-channel tensors: each is (1, H_r, W_r). Concatenate along channel dim.
        # Resulting shape (n_frames_input, H_r, W_r), then add batch dim.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): 
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Use fastai Image for resizing tensor
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of 16 and at least 16
        new_w = max(16, new_w - (new_w % 16))
        new_h = max(16, new_h - (new_h % 16))
        return new_w, new_h

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        t_l_scaled = t_l * 100.0 # Scale L channel to [0, 100]
        return torch.cat([t_l_scaled, t_ab], dim=0) # Result (3, H, W)


# Imports for VideoColorizerFilter are mostly covered by existing imports.
# torch, Tensor, nn, Image (FastAI), vision_transform, List, Tuple, logging, PilImage are already available
# via existing `from fastai.vision import *` and `from fastai.core import *`
from .augs import denorm_lab_mean_std_tensor # Specific import needed
# IFilter is defined in this file. nn.Module is available from fastai.vision too.


class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, **kwargs):
        super().__init__(**kwargs) 
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        try:
            self.device = next(model.parameters()).device
        except StopIteration: # Model has no parameters
            self.device = torch.device("cpu") 
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")


    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        # This method processes a single image by duplicating it n_frames_input times
        # to form a sequence, then calling process_sequence.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor=render_factor, post_process=post_process)
    
    # Required by IFilter interface
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # We process 'filtered_image' as it's the output from the previous stage (or the input if this is the first filter).
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 16 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 16 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)

        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack L-channel tensors: each is (1, H_r, W_r). Concatenate along channel dim.
        # Resulting shape (n_frames_input, H_r, W_r), then add batch dim.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): 
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Use fastai Image for resizing tensor
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of 16 and at least 16
        new_w = max(16, new_w - (new_w % 16))
        new_h = max(16, new_h - (new_h % 16))
        return new_w, new_h

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        t_l_scaled = t_l * 100.0 # Scale L channel to [0, 100]
        return torch.cat([t_l_scaled, t_ab], dim=0) # Result (3, H, W)


# Imports for VideoColorizerFilter are mostly covered by existing imports.
# torch, Tensor, nn, Image (FastAI), vision_transform, List, Tuple, logging, PilImage are already available
# via existing `from fastai.vision import *` and `from fastai.core import *`
from .augs import denorm_lab_mean_std_tensor # Specific import needed
# IFilter is defined in this file. nn.Module is available from fastai.vision too.


class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, **kwargs):
        super().__init__(**kwargs) 
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        try:
            self.device = next(model.parameters()).device
        except StopIteration: # Model has no parameters
            self.device = torch.device("cpu") 
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")


    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        # This method processes a single image by duplicating it n_frames_input times
        # to form a sequence, then calling process_sequence.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor=render_factor, post_process=post_process)
    
    # Required by IFilter interface
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # We process 'filtered_image' as it's the output from the previous stage (or the input if this is the first filter).
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 16 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 16 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)

        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack L-channel tensors: each is (1, H_r, W_r). Concatenate along channel dim.
        # Resulting shape (n_frames_input, H_r, W_r), then add batch dim.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): 
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Use fastai Image for resizing tensor
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of 16 and at least 16
        new_w = max(16, new_w - (new_w % 16))
        new_h = max(16, new_h - (new_h % 16))
        return new_w, new_h

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        t_l_scaled = t_l * 100.0 # Scale L channel to [0, 100]
        return torch.cat([t_l_scaled, t_ab], dim=0) # Result (3, H, W)


# Imports for VideoColorizerFilter are mostly covered by existing imports.
# torch, Tensor, nn, Image (FastAI), vision.transform, List, Tuple, logging, PilImage are already available
# via existing `from fastai.vision import *` and `from fastai.core import *`
from .augs import denorm_lab_mean_std_tensor # Specific import needed
# IFilter is defined in this file. nn.Module is available from fastai.vision too.


class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, **kwargs):
        super().__init__(**kwargs) 
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        try:
            self.device = next(model.parameters()).device
        except StopIteration: # Model has no parameters
            self.device = torch.device("cpu") 
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")


    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        # This method processes a single image by duplicating it n_frames_input times
        # to form a sequence, then calling process_sequence.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor=render_factor, post_process=post_process)
    
    # Required by IFilter interface
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # We process 'filtered_image' as it's the output from the previous stage (or the input if this is the first filter).
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 16 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 16 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)

        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack L-channel tensors: each is (1, H_r, W_r). Concatenate along channel dim.
        # Resulting shape (n_frames_input, H_r, W_r), then add batch dim.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): 
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Use fastai Image for resizing tensor
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of 16 and at least 16
        new_w = max(16, new_w - (new_w % 16))
        new_h = max(16, new_h - (new_h % 16))
        return new_w, new_h

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        t_l_scaled = t_l * 100.0 # Scale L channel to [0, 100]
        return torch.cat([t_l_scaled, t_ab], dim=0) # Result (3, H, W)


# Imports for VideoColorizerFilter are mostly covered by existing imports.
# torch, Tensor, nn, Image (FastAI), vision_transform, List, Tuple, logging, PilImage are already available
# via existing `from fastai.vision import *` and `from fastai.core import *`
from .augs import denorm_lab_mean_std_tensor # Specific import needed
# IFilter is defined in this file. nn.Module is available from fastai.vision too.


class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, **kwargs):
        super().__init__(**kwargs) 
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        try:
            self.device = next(model.parameters()).device
        except StopIteration: # Model has no parameters
            self.device = torch.device("cpu") 
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")


    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        # This method processes a single image by duplicating it n_frames_input times
        # to form a sequence, then calling process_sequence.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor=render_factor, post_process=post_process)
    
    # Required by IFilter interface
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # We process 'filtered_image' as it's the output from the previous stage (or the input if this is the first filter).
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 16 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 16 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)

        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack L-channel tensors: each is (1, H_r, W_r). Concatenate along channel dim.
        # Resulting shape (n_frames_input, H_r, W_r), then add batch dim.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): 
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Use fastai Image for resizing tensor
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of 16 and at least 16
        new_w = max(16, new_w - (new_w % 16))
        new_h = max(16, new_h - (new_h % 16))
        return new_w, new_h

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        t_l_scaled = t_l * 100.0 # Scale L channel to [0, 100]
        return torch.cat([t_l_scaled, t_ab], dim=0) # Result (3, H, W)


# Imports for VideoColorizerFilter are mostly covered by existing imports.
# torch, Tensor, nn, Image (FastAI), vision_transform, List, Tuple, logging, PilImage are already available
# via existing `from fastai.vision import *` and `from fastai.core import *`
from .augs import denorm_lab_mean_std_tensor # Specific import needed
# IFilter is defined in this file. nn.Module is available from fastai.vision too.


class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, **kwargs):
        super().__init__(**kwargs) 
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        try:
            self.device = next(model.parameters()).device
        except StopIteration: # Model has no parameters
            self.device = torch.device("cpu") 
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")


    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        # This method processes a single image by duplicating it n_frames_input times
        # to form a sequence, then calling process_sequence.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor=render_factor, post_process=post_process)
    
    # Required by IFilter interface
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # We process 'filtered_image' as it's the output from the previous stage (or the input if this is the first filter).
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 16 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 16 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)

        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack L-channel tensors: each is (1, H_r, W_r). Concatenate along channel dim.
        # Resulting shape (n_frames_input, H_r, W_r), then add batch dim.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): 
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Use fastai Image for resizing tensor
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of 16 and at least 16
        new_w = max(16, new_w - (new_w % 16))
        new_h = max(16, new_h - (new_h % 16))
        return new_w, new_h

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        t_l_scaled = t_l * 100.0 # Scale L channel to [0, 100]
        return torch.cat([t_l_scaled, t_ab], dim=0) # Result (3, H, W)


# Imports for VideoColorizerFilter are mostly covered by existing imports.
# torch, Tensor, nn, Image (FastAI), vision_transform, List, Tuple, logging, PilImage are already available
# via existing `from fastai.vision import *` and `from fastai.core import *`
from .augs import denorm_lab_mean_std_tensor # Specific import needed
# IFilter is defined in this file. nn.Module is available from fastai.vision too.


class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, **kwargs):
        super().__init__(**kwargs) 
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        try:
            self.device = next(model.parameters()).device
        except StopIteration: # Model has no parameters
            self.device = torch.device("cpu") 
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")


    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        # This method processes a single image by duplicating it n_frames_input times
        # to form a sequence, then calling process_sequence.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor=render_factor, post_process=post_process)
    
    # Required by IFilter interface
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # We process 'filtered_image' as it's the output from the previous stage (or the input if this is the first filter).
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 16 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 16 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)

        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack L-channel tensors: each is (1, H_r, W_r). Concatenate along channel dim.
        # Resulting shape (n_frames_input, H_r, W_r), then add batch dim.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): 
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Use fastai Image for resizing tensor
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of 16 and at least 16
        new_w = max(16, new_w - (new_w % 16))
        new_h = max(16, new_h - (new_h % 16))
        return new_w, new_h

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        t_l_scaled = t_l * 100.0 # Scale L channel to [0, 100]
        return torch.cat([t_l_scaled, t_ab], dim=0) # Result (3, H, W)


# Imports for VideoColorizerFilter are mostly covered by existing imports.
# torch, Tensor, nn, Image (FastAI), vision_transform, List, Tuple, logging, PilImage are already available
# via existing `from fastai.vision import *` and `from fastai.core import *`
from .augs import denorm_lab_mean_std_tensor # Specific import needed
# IFilter is defined in this file. nn.Module is available from fastai.vision too.


class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, **kwargs):
        super().__init__(**kwargs) 
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        try:
            self.device = next(model.parameters()).device
        except StopIteration: # Model has no parameters
            self.device = torch.device("cpu") 
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")


    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        # This method processes a single image by duplicating it n_frames_input times
        # to form a sequence, then calling process_sequence.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor=render_factor, post_process=post_process)
    
    # Required by IFilter interface
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # We process 'filtered_image' as it's the output from the previous stage (or the input if this is the first filter).
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 16 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 16 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)

        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack L-channel tensors: each is (1, H_r, W_r). Concatenate along channel dim.
        # Resulting shape (n_frames_input, H_r, W_r), then add batch dim.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): 
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Use fastai Image for resizing tensor
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of 16 and at least 16
        new_w = max(16, new_w - (new_w % 16))
        new_h = max(16, new_h - (new_h % 16))
        return new_w, new_h

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        t_l_scaled = t_l * 100.0 # Scale L channel to [0, 100]
        return torch.cat([t_l_scaled, t_ab], dim=0) # Result (3, H, W)


# Imports for VideoColorizerFilter are mostly covered by existing imports.
# torch, Tensor, nn, Image (FastAI), vision.transform, List, Tuple, logging, PilImage are already available
# via existing `from fastai.vision import *` and `from fastai.core import *`
from .augs import denorm_lab_mean_std_tensor # Specific import needed
# IFilter is defined in this file. nn.Module is available from fastai.vision too.


class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, **kwargs):
        super().__init__(**kwargs) 
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        try:
            self.device = next(model.parameters()).device
        except StopIteration: # Model has no parameters
            self.device = torch.device("cpu") 
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")


    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        # This method processes a single image by duplicating it n_frames_input times
        # to form a sequence, then calling process_sequence.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor=render_factor, post_process=post_process)
    
    # Required by IFilter interface
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # We process 'filtered_image' as it's the output from the previous stage (or the input if this is the first filter).
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 16 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 16 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)

        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack L-channel tensors: each is (1, H_r, W_r). Concatenate along channel dim.
        # Resulting shape (n_frames_input, H_r, W_r), then add batch dim.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): 
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Use fastai Image for resizing tensor
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of 16 and at least 16
        new_w = max(16, new_w - (new_w % 16))
        new_h = max(16, new_h - (new_h % 16))
        return new_w, new_h

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        t_l_scaled = t_l * 100.0 # Scale L channel to [0, 100]
        return torch.cat([t_l_scaled, t_ab], dim=0) # Result (3, H, W)


# Imports for VideoColorizerFilter
# torch, Tensor, nn, Image (FastAI), vision_transform, List, Tuple, logging, PilImage are already available
# via existing `from fastai.vision import *` and `from fastai.core import *`
from .augs import denorm_lab_mean_std_tensor # Specific import needed
# IFilter is defined in this file. nn.Module is available from fastai.vision too.


class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, **kwargs):
        super().__init__(**kwargs) 
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        try:
            self.device = next(model.parameters()).device
        except StopIteration: # Model has no parameters
            self.device = torch.device("cpu") 
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")


    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        # This method processes a single image by duplicating it n_frames_input times
        # to form a sequence, then calling process_sequence.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor=render_factor, post_process=post_process)
    
    # Required by IFilter interface
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # We process 'filtered_image' as it's the output from the previous stage (or the input if this is the first filter).
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 16 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 16 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)

        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack L-channel tensors: each is (1, H_r, W_r). Concatenate along channel dim.
        # Resulting shape (n_frames_input, H_r, W_r), then add batch dim.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): # Corrected from torch.no_cache()
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Use fastai Image for resizing tensor
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of 16 and at least 16
        new_w = max(16, new_w - (new_w % 16))
        new_h = max(16, new_h - (new_h % 16))
        return new_w, new_h

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        t_l_scaled = t_l * 100.0 # Scale L channel to [0, 100]
        return torch.cat([t_l_scaled, t_ab], dim=0) # Result (3, H, W)


# Imports for VideoColorizerFilter
# torch, Tensor, nn, Image (FastAI), vision_transform, List, Tuple, logging, PilImage are already available
# via existing `from fastai.vision import *` and `from fastai.core import *`
from .augs import denorm_lab_mean_std_tensor # Specific import needed
# IFilter is defined in this file. nn.Module is available from fastai.vision too.


class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, **kwargs):
        super().__init__(**kwargs) 
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        try:
            self.device = next(model.parameters()).device
        except StopIteration: # Model has no parameters
            self.device = torch.device("cpu") 
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")


    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        # This method processes a single image by duplicating it n_frames_input times
        # to form a sequence, then calling process_sequence.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor=render_factor, post_process=post_process)
    
    # Required by IFilter interface
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # We process 'filtered_image' as it's the output from the previous stage (or the input if this is the first filter).
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 16 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 16 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)

        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack L-channel tensors: each is (1, H_r, W_r). Concatenate along channel dim.
        # Resulting shape (n_frames_input, H_r, W_r), then add batch dim.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): # Corrected from torch.no_cache()
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Use fastai Image for resizing tensor
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of 16 and at least 16
        new_w = max(16, new_w - (new_w % 16))
        new_h = max(16, new_h - (new_h % 16))
        return new_w, new_h

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        t_l_scaled = t_l * 100.0 # Scale L channel to [0, 100]
        return torch.cat([t_l_scaled, t_ab], dim=0) # Result (3, H, W)


# Imports for VideoColorizerFilter
# torch, Tensor, nn, Image (FastAI), vision.transform, List, Tuple, logging, PilImage are already available
# via existing `from fastai.vision import *` and `from fastai.core import *`
from .augs import denorm_lab_mean_std_tensor
# IFilter is defined in this file. nn.Module is available from fastai.vision too.


class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, **kwargs):
        super().__init__(**kwargs) 
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        try:
            self.device = next(model.parameters()).device
        except StopIteration: # Model has no parameters
            self.device = torch.device("cpu") 
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")


    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        # This method processes a single image by duplicating it n_frames_input times
        # to form a sequence, then calling process_sequence.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor=render_factor, post_process=post_process)
    
    # Required by IFilter interface
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # We process 'filtered_image' as it's the output from the previous stage (or the input if this is the first filter).
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 16 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 16 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)

        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack L-channel tensors: each is (1, H_r, W_r). Concatenate along channel dim.
        # Resulting shape (n_frames_input, H_r, W_r), then add batch dim.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Resize ab_output_tensor to match original middle frame's size
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of 16 and at least 16
        new_w = max(16, new_w - (new_w % 16))
        new_h = max(16, new_h - (new_h % 16))
        return new_w, new_h

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        t_l_scaled = t_l * 100.0 # Scale L channel to [0, 100]
        return torch.cat([t_l_scaled, t_ab], dim=0) # Result (3, H, W)


# Imports for VideoColorizerFilter are mostly covered by existing imports.
# torch, Tensor, nn, Image (FastAI), vision.transform, List, Tuple, logging, PilImage are already available.
from .augs import denorm_lab_mean_std_tensor # Specific import needed
# IFilter is defined in this file.

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
        # This method is for single images (IFilter compatibility), 
        # processed by duplicating the frame n_frames_input times.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor, post_process)
    
    # Required by IFilter
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline (used for context if needed, not directly here).
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 0 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 0 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)


        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): # Corrected from torch.no_cache()
            ab_output_tensor = self.model(input_tensor).squeeze(0) 

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Use fastai Image for resizing tensor
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
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


# Imports for VideoColorizerFilter are mostly covered by existing imports.
# torch, Tensor, nn, Image (FastAI), vision.transform, List, Tuple, logging, PilImage are already available.
from .augs import denorm_lab_mean_std_tensor # Specific import needed
# IFilter is defined in this file.

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
        # This method is for single images (IFilter compatibility), 
        # processed by duplicating the frame n_frames_input times.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor, post_process)
    
    # Required by IFilter
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline (used for context if needed, not directly here).
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 0 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 0 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)


        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): # Corrected from torch.no_cache()
            ab_output_tensor = self.model(input_tensor).squeeze(0) 

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Use fastai Image for resizing tensor
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
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


# Imports for VideoColorizerFilter are mostly covered by existing imports.
# torch, Tensor, nn, Image (FastAI), vision.transform, List, Tuple, logging, PilImage are already available.
from .augs import denorm_lab_mean_std_tensor # Specific import needed
# IFilter is defined in this file.

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
    
    # Required by IFilter
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
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
        else: # Ensure dimensions are divisible by 16 if no render_factor
            target_w_render = target_w_render - (target_w_render % 16) if target_w_render > 0 else 16
            target_h_render = target_h_render - (target_h_render % 16) if target_h_render > 0 else 16
            target_w_render = max(16, target_w_render)
            target_h_render = max(16, target_h_render)


        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad(): # Corrected from torch.no_cache()
            ab_output_tensor = self.model(input_tensor).squeeze(0) 

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Use fastai Image for resizing tensor
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
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


# Imports for VideoColorizerFilter
# torch, Tensor, nn are already available via fastai.vision import *
# PilImage is already imported as from PIL import Image as PilImage
# Image (FastAI) is available via from fastai.vision.image import *
# vision.transform is available via from fastai.vision import *
# List, Tuple are available via from fastai.core import *
from .augs import denorm_lab_mean_std_tensor
# IFilter is defined in this file, logging is already imported.


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
        # This method is for single images (IFilter compatibility), 
        # processed by duplicating the frame n_frames_input times.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor, post_process)
    
    # Required by IFilter
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline (used for context if needed, not directly here).
        return self.P(filtered_image, render_factor=render_factor, post_process=post_process)


    def process_sequence(self, frame_sequence: List[PilImage], render_factor: int = None, post_process: bool = True) -> PilImage:
        if len(frame_sequence) != self.n_frames_input:
            raise ValueError(f"Expected {self.n_frames_input} frames, got {len(frame_sequence)}")

        current_render_factor = render_factor if render_factor is not None else self.render_factor
        
        processed_frames_l_tensors = [] 
        middle_frame_idx = self.n_frames_input // 2
        orig_middle_frame_pil = frame_sequence[middle_frame_idx]
        
        target_w_render, target_h_render = orig_middle_frame_pil.size # Default if no render_factor
        if current_render_factor is not None:
            target_w_render, target_h_render = self.get_render_factor_target_size(orig_middle_frame_pil, current_render_factor)
        else:
            # Ensure divisible by 16 if no render factor
            target_w_render = target_w_render - target_w_render % 16
            target_h_render = target_h_render - target_h_render % 16
            if target_w_render == 0 or target_h_render == 0: # Handle very small images
                 target_w_render, target_h_render = orig_middle_frame_pil.size


        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            # vision.transform.ToTensor() is already available from `from fastai.vision import *`
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack frames for the model input: (C, H, W) -> (n_frames_input * C, H, W)
        # Assuming C=1 for L-channel images.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) # Shape: (1, n_frames_input, H_r, W_r)
        input_tensor = input_tensor.to(self.device)

        # Model inference
        with torch.no_cache():
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        # Post-processing
        # Get original L channel of the middle frame
        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Resize ab_output_tensor to match original middle frame's size
        # FastAI's Image class handles tensor resizing well.
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) # (1, H_orig, W_orig)
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of a number if model requires (e.g., 16)
        new_w = new_w - new_w % 16 if new_w > 0 else 16 # Ensure at least 16
        new_h = new_h - new_h % 16 if new_h > 0 else 16 # Ensure at least 16
        return max(16, new_w), max(16, new_h) 

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        # Standard LAB: L [0,100], A [-128,127], B [-128,127]
        t_l_scaled = t_l * 100.0 
        return torch.cat([t_l_scaled, t_ab], dim=0) # (3, H, W)


# Imports for VideoColorizerFilter
# torch, Tensor, nn are already available via fastai.vision import *
# PilImage is already imported as from PIL import Image as PilImage
# Image (FastAI) is available via from fastai.vision.image import *
# vision.transform is available via from fastai.vision import *
# List, Tuple are available via from fastai.core import *
from .augs import denorm_lab_mean_std_tensor
# IFilter is defined in this file, logging is already imported.


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
        # This method is for single images (IFilter compatibility), 
        # processed by duplicating the frame n_frames_input times.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor, post_process)
    
    # Required by IFilter
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline (used for context if needed, not directly here).
        return self.P(filtered_image, render_factor=render_factor, post_process=post_process)


    def process_sequence(self, frame_sequence: List[PilImage], render_factor: int = None, post_process: bool = True) -> PilImage:
        if len(frame_sequence) != self.n_frames_input:
            raise ValueError(f"Expected {self.n_frames_input} frames, got {len(frame_sequence)}")

        current_render_factor = render_factor if render_factor is not None else self.render_factor
        
        processed_frames_l_tensors = [] 
        middle_frame_idx = self.n_frames_input // 2
        orig_middle_frame_pil = frame_sequence[middle_frame_idx]
        
        target_w_render, target_h_render = orig_middle_frame_pil.size # Default if no render_factor
        if current_render_factor is not None:
            target_w_render, target_h_render = self.get_render_factor_target_size(orig_middle_frame_pil, current_render_factor)
        else:
            # Ensure divisible by 16 if no render factor
            target_w_render = target_w_render - target_w_render % 16
            target_h_render = target_h_render - target_h_render % 16
            if target_w_render == 0 or target_h_render == 0: # Handle very small images
                 target_w_render, target_h_render = orig_middle_frame_pil.size


        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            # vision.transform.ToTensor() is already available from `from fastai.vision import *`
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack frames for the model input: (C, H, W) -> (n_frames_input * C, H, W)
        # Assuming C=1 for L-channel images.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) # Shape: (1, n_frames_input, H_r, W_r)
        input_tensor = input_tensor.to(self.device)

        # Model inference
        with torch.no_cache():
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        # Post-processing
        # Get original L channel of the middle frame
        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Resize ab_output_tensor to match original middle frame's size
        # FastAI's Image class handles tensor resizing well.
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) # (1, H_orig, W_orig)
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of a number if model requires (e.g., 16)
        new_w = new_w - new_w % 16 if new_w > 0 else 16 # Ensure at least 16
        new_h = new_h - new_h % 16 if new_h > 0 else 16 # Ensure at least 16
        return max(16, new_w), max(16, new_h) 

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        # Standard LAB: L [0,100], A [-128,127], B [-128,127]
        t_l_scaled = t_l * 100.0 
        return torch.cat([t_l_scaled, t_ab], dim=0) # (3, H, W)


# Imports for VideoColorizerFilter
# torch, Tensor, nn are already available via fastai.vision import *
# PilImage is already imported as from PIL import Image as PilImage
# Image (FastAI) is available via from fastai.vision.image import *
# vision.transform is available via from fastai.vision import *
# List, Tuple are available via from fastai.core import *
from .augs import denorm_lab_mean_std_tensor
# IFilter is defined in this file, logging is already imported.


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
        # This method is for single images (IFilter compatibility), 
        # processed by duplicating the frame n_frames_input times.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor, post_process)
    
    # Required by IFilter
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline (used for context if needed, not directly here).
        return self.P(filtered_image, render_factor=render_factor, post_process=post_process)


    def process_sequence(self, frame_sequence: List[PilImage], render_factor: int = None, post_process: bool = True) -> PilImage:
        if len(frame_sequence) != self.n_frames_input:
            raise ValueError(f"Expected {self.n_frames_input} frames, got {len(frame_sequence)}")

        current_render_factor = render_factor if render_factor is not None else self.render_factor
        
        processed_frames_l_tensors = [] 
        middle_frame_idx = self.n_frames_input // 2
        orig_middle_frame_pil = frame_sequence[middle_frame_idx]
        
        target_w_render, target_h_render = orig_middle_frame_pil.size # Default if no render_factor
        if current_render_factor is not None:
            target_w_render, target_h_render = self.get_render_factor_target_size(orig_middle_frame_pil, current_render_factor)
        else:
            # Ensure divisible by 16 if no render factor
            target_w_render = target_w_render - target_w_render % 16
            target_h_render = target_h_render - target_h_render % 16
            if target_w_render == 0 or target_h_render == 0: # Handle very small images
                 target_w_render, target_h_render = orig_middle_frame_pil.size


        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            # vision.transform.ToTensor() is already available from `from fastai.vision import *`
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack frames for the model input: (C, H, W) -> (n_frames_input * C, H, W)
        # Assuming C=1 for L-channel images.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) # Shape: (1, n_frames_input, H_r, W_r)
        input_tensor = input_tensor.to(self.device)

        # Model inference
        with torch.no_cache():
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        # Post-processing
        # Get original L channel of the middle frame
        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Resize ab_output_tensor to match original middle frame's size
        # FastAI's Image class handles tensor resizing well.
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) # (1, H_orig, W_orig)
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of a number if model requires (e.g., 16)
        new_w = new_w - new_w % 16 if new_w > 0 else 16 # Ensure at least 16
        new_h = new_h - new_h % 16 if new_h > 0 else 16 # Ensure at least 16
        return max(16, new_w), max(16, new_h) 

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        # Standard LAB: L [0,100], A [-128,127], B [-128,127]
        t_l_scaled = t_l * 100.0 
        return torch.cat([t_l_scaled, t_ab], dim=0) # (3, H, W)


# Imports for VideoColorizerFilter
# torch, Tensor, nn are already available via fastai.vision import *
# PilImage is already imported as from PIL import Image as PilImage
# Image (FastAI) is available via from fastai.vision.image import *
# vision.transform is available via from fastai.vision import *
# List, Tuple are available via from fastai.core import *
from .augs import denorm_lab_mean_std_tensor
# IFilter is defined in this file, logging is already imported.


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
        # This method is for single images (IFilter compatibility), 
        # processed by duplicating the frame n_frames_input times.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor, post_process)
    
    # Required by IFilter
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline (used for context if needed, not directly here).
        return self.P(filtered_image, render_factor=render_factor, post_process=post_process)


    def process_sequence(self, frame_sequence: List[PilImage], render_factor: int = None, post_process: bool = True) -> PilImage:
        if len(frame_sequence) != self.n_frames_input:
            raise ValueError(f"Expected {self.n_frames_input} frames, got {len(frame_sequence)}")

        current_render_factor = render_factor if render_factor is not None else self.render_factor
        
        processed_frames_l_tensors = [] 
        middle_frame_idx = self.n_frames_input // 2
        orig_middle_frame_pil = frame_sequence[middle_frame_idx]
        
        target_w_render, target_h_render = orig_middle_frame_pil.size # Default if no render_factor
        if current_render_factor is not None:
            target_w_render, target_h_render = self.get_render_factor_target_size(orig_middle_frame_pil, current_render_factor)
        else:
            # Ensure divisible by 16 if no render factor
            target_w_render = target_w_render - target_w_render % 16
            target_h_render = target_h_render - target_h_render % 16
            if target_w_render == 0 or target_h_render == 0: # Handle very small images
                 target_w_render, target_h_render = orig_middle_frame_pil.size


        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            # vision.transform.ToTensor() is already available from `from fastai.vision import *`
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack frames for the model input: (C, H, W) -> (n_frames_input * C, H, W)
        # Assuming C=1 for L-channel images.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) # Shape: (1, n_frames_input, H_r, W_r)
        input_tensor = input_tensor.to(self.device)

        # Model inference
        with torch.no_cache():
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        # Post-processing
        # Get original L channel of the middle frame
        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Resize ab_output_tensor to match original middle frame's size
        # FastAI's Image class handles tensor resizing well.
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) # (1, H_orig, W_orig)
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of a number if model requires (e.g., 16)
        new_w = new_w - new_w % 16 if new_w > 0 else 16 # Ensure at least 16
        new_h = new_h - new_h % 16 if new_h > 0 else 16 # Ensure at least 16
        return max(16, new_w), max(16, new_h) 

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        # Standard LAB: L [0,100], A [-128,127], B [-128,127]
        t_l_scaled = t_l * 100.0 
        return torch.cat([t_l_scaled, t_ab], dim=0) # (3, H, W)


# Imports for VideoColorizerFilter
# torch, Tensor, nn are already available via fastai.vision import *
# PilImage is already imported as from PIL import Image as PilImage
# Image (FastAI) is available via from fastai.vision.image import *
# vision.transform is available via from fastai.vision import *
# List, Tuple are available via from fastai.core import *
from .augs import denorm_lab_mean_std_tensor
# IFilter is defined in this file, logging is already imported.


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
        # This method is for single images (IFilter compatibility), 
        # processed by duplicating the frame n_frames_input times.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor, post_process)
    
    # Required by IFilter
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline (used for context if needed, not directly here).
        return self.P(filtered_image, render_factor=render_factor, post_process=post_process)


    def process_sequence(self, frame_sequence: List[PilImage], render_factor: int = None, post_process: bool = True) -> PilImage:
        if len(frame_sequence) != self.n_frames_input:
            raise ValueError(f"Expected {self.n_frames_input} frames, got {len(frame_sequence)}")

        current_render_factor = render_factor if render_factor is not None else self.render_factor
        
        processed_frames_l_tensors = [] 
        middle_frame_idx = self.n_frames_input // 2
        orig_middle_frame_pil = frame_sequence[middle_frame_idx]
        
        target_w_render, target_h_render = orig_middle_frame_pil.size # Default if no render_factor
        if current_render_factor is not None:
            target_w_render, target_h_render = self.get_render_factor_target_size(orig_middle_frame_pil, current_render_factor)
        else:
            # Ensure divisible by 16 if no render factor
            target_w_render = target_w_render - target_w_render % 16
            target_h_render = target_h_render - target_h_render % 16
            if target_w_render == 0 or target_h_render == 0: # Handle very small images
                 target_w_render, target_h_render = orig_middle_frame_pil.size


        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            # vision.transform.ToTensor() is already available from `from fastai.vision import *`
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack frames for the model input: (C, H, W) -> (n_frames_input * C, H, W)
        # Assuming C=1 for L-channel images.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) # Shape: (1, n_frames_input, H_r, W_r)
        input_tensor = input_tensor.to(self.device)

        # Model inference
        with torch.no_cache():
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        # Post-processing
        # Get original L channel of the middle frame
        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Resize ab_output_tensor to match original middle frame's size
        # FastAI's Image class handles tensor resizing well.
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) # (1, H_orig, W_orig)
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of a number if model requires (e.g., 16)
        new_w = new_w - new_w % 16 if new_w > 0 else 16 # Ensure at least 16
        new_h = new_h - new_h % 16 if new_h > 0 else 16 # Ensure at least 16
        return max(16, new_w), max(16, new_h) 

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        # Standard LAB: L [0,100], A [-128,127], B [-128,127]
        t_l_scaled = t_l * 100.0 
        return torch.cat([t_l_scaled, t_ab], dim=0) # (3, H, W)


# Imports for VideoColorizerFilter
# torch, Tensor, nn are already available via fastai.vision import *
# PilImage is already imported as from PIL import Image as PilImage
# Image (FastAI) is available via from fastai.vision.image import *
# vision.transform is available via from fastai.vision import *
# List, Tuple are available via from fastai.core import *
from .augs import denorm_lab_mean_std_tensor
# IFilter is defined in this file, logging is already imported.


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
        # This method is for single images (IFilter compatibility), 
        # processed by duplicating the frame n_frames_input times.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor, post_process)
    
    # Required by IFilter
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline (used for context if needed, not directly here).
        return self.P(filtered_image, render_factor=render_factor, post_process=post_process)


    def process_sequence(self, frame_sequence: List[PilImage], render_factor: int = None, post_process: bool = True) -> PilImage:
        if len(frame_sequence) != self.n_frames_input:
            raise ValueError(f"Expected {self.n_frames_input} frames, got {len(frame_sequence)}")

        current_render_factor = render_factor if render_factor is not None else self.render_factor
        
        processed_frames_l_tensors = [] 
        middle_frame_idx = self.n_frames_input // 2
        orig_middle_frame_pil = frame_sequence[middle_frame_idx]
        
        target_w_render, target_h_render = orig_middle_frame_pil.size # Default if no render_factor
        if current_render_factor is not None:
            target_w_render, target_h_render = self.get_render_factor_target_size(orig_middle_frame_pil, current_render_factor)
        else:
            # Ensure divisible by 16 if no render factor
            target_w_render = target_w_render - target_w_render % 16
            target_h_render = target_h_render - target_h_render % 16
            if target_w_render == 0 or target_h_render == 0: # Handle very small images
                 target_w_render, target_h_render = orig_middle_frame_pil.size


        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            # vision.transform.ToTensor() is already available from `from fastai.vision import *`
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack frames for the model input: (C, H, W) -> (n_frames_input * C, H, W)
        # Assuming C=1 for L-channel images.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) # Shape: (1, n_frames_input, H_r, W_r)
        input_tensor = input_tensor.to(self.device)

        # Model inference
        with torch.no_cache():
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        # Post-processing
        # Get original L channel of the middle frame
        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Resize ab_output_tensor to match original middle frame's size
        # FastAI's Image class handles tensor resizing well.
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) # (1, H_orig, W_orig)
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of a number if model requires (e.g., 16)
        new_w = new_w - new_w % 16 if new_w > 0 else 16 # Ensure at least 16
        new_h = new_h - new_h % 16 if new_h > 0 else 16 # Ensure at least 16
        return max(16, new_w), max(16, new_h) 

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        # Standard LAB: L [0,100], A [-128,127], B [-128,127]
        t_l_scaled = t_l * 100.0 
        return torch.cat([t_l_scaled, t_ab], dim=0) # (3, H, W)


# Imports for VideoColorizerFilter
# torch, Tensor, nn are available via fastai.vision import *
# PilImage is already imported as from PIL import Image as PilImage
# Image (FastAI) is available via from fastai.vision.image import *
# vision.transform is available via from fastai.vision import *
# List, Tuple are available via from fastai.core import *
from .augs import denorm_lab_mean_std_tensor
# IFilter is defined in this file, logging is already imported.


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
        # This method is for single images (IFilter compatibility), 
        # processed by duplicating the frame n_frames_input times.
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor, post_process)
    
    # Required by IFilter
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline (used for context if needed, not directly here).
        return self.P(filtered_image, render_factor=render_factor, post_process=post_process)


    def process_sequence(self, frame_sequence: List[PilImage], render_factor: int = None, post_process: bool = True) -> PilImage:
        if len(frame_sequence) != self.n_frames_input:
            raise ValueError(f"Expected {self.n_frames_input} frames, got {len(frame_sequence)}")

        current_render_factor = render_factor if render_factor is not None else self.render_factor
        
        processed_frames_l_tensors = [] 
        middle_frame_idx = self.n_frames_input // 2
        orig_middle_frame_pil = frame_sequence[middle_frame_idx]
        
        target_w_render, target_h_render = orig_middle_frame_pil.size # Default if no render_factor
        if current_render_factor is not None:
            target_w_render, target_h_render = self.get_render_factor_target_size(orig_middle_frame_pil, current_render_factor)
        else:
            # Ensure divisible by 16 if no render factor
            target_w_render = target_w_render - target_w_render % 16
            target_h_render = target_h_render - target_h_render % 16
            if target_w_render == 0 or target_h_render == 0: # Handle very small images
                 target_w_render, target_h_render = orig_middle_frame_pil.size


        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            # vision.transform.ToTensor() is already available from `from fastai.vision import *`
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        # Stack frames for the model input: (C, H, W) -> (n_frames_input * C, H, W)
        # Assuming C=1 for L-channel images.
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) # Shape: (1, n_frames_input, H_r, W_r)
        input_tensor = input_tensor.to(self.device)

        # Model inference
        with torch.no_cache():
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected model output: (2, H_r, W_r)

        # Post-processing
        # Get original L channel of the middle frame
        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Resize ab_output_tensor to match original middle frame's size
        # FastAI's Image class handles tensor resizing well.
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) # (1, H_orig, W_orig)
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of a number if model requires (e.g., 16)
        new_w = new_w - new_w % 16 if new_w > 0 else 16 # Ensure at least 16
        new_h = new_h - new_h % 16 if new_h > 0 else 16 # Ensure at least 16
        return max(16, new_w), max(16, new_h) 

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model
        # Standard LAB: L [0,100], A [-128,127], B [-128,127]
        t_l_scaled = t_l * 100.0 
        return torch.cat([t_l_scaled, t_ab], dim=0) # (3, H, W)


# Imports for VideoColorizerFilter
import torch
from torch import Tensor # Already imported via fastai.torch_core
import torch.nn as nn # Already imported via fastai.vision
# PilImage is already imported as from PIL import Image as PilImage
# from fastai.vision.image import Image is already imported via from fastai.vision.image import *
# from fastai.vision import transform as vision_transform # transform is available via from fastai.vision import *
# from typing import List, Tuple # Already imported via fastai.core -> from .imports.core import *
from .augs import denorm_lab_mean_std_tensor
# IFilter is defined in this file


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
    
    # Required by IFilter
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # For this filter, we assume it's the primary colorization step, so it works on 'filtered_image'.
        return self.P(filtered_image, render_factor=render_factor, post_process=post_process)


    def process_sequence(self, frame_sequence: List[PilImage], render_factor: int = None, post_process: bool = True) -> PilImage:
        if len(frame_sequence) != self.n_frames_input:
            raise ValueError(f"Expected {self.n_frames_input} frames, got {len(frame_sequence)}")

        current_render_factor = render_factor if render_factor is not None else self.render_factor
        
        processed_frames_l_tensors = [] 
        middle_frame_idx = self.n_frames_input // 2
        orig_middle_frame_pil = frame_sequence[middle_frame_idx]
        
        target_w_render, target_h_render = orig_middle_frame_pil.size # Default if no render_factor
        if current_render_factor is not None:
            target_w_render, target_h_render = self.get_render_factor_target_size(orig_middle_frame_pil, current_render_factor)
        else:
            # Ensure divisible by 16 if no render factor
            target_w_render = target_w_render - target_w_render % 16
            target_h_render = target_h_render - target_h_render % 16
            if target_w_render == 0 or target_h_render == 0:
                 target_w_render, target_h_render = orig_middle_frame_pil.size


        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_cache():
            ab_output_tensor = self.model(input_tensor).squeeze(0) 

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Resize ab_output_tensor to match original middle frame's size
        # Use FastAIImage for its resize method that works on tensors directly
        # Note: fastai.vision.image.Image is typically aliased as `Image` in fastai
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
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


# Imports for VideoColorizerFilter
import torch
from torch import Tensor # Already imported via fastai.torch_core
import torch.nn as nn # Already imported via fastai.vision
# PilImage is already imported as from PIL import Image as PilImage
# from fastai.vision.image import Image is already imported via from fastai.vision.image import *
# from fastai.vision import transform as vision_transform # transform is available via from fastai.vision import *
# from typing import List, Tuple # Already imported via fastai.core -> from .imports.core import *
from .augs import denorm_lab_mean_std_tensor
# IFilter is defined in this file


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
    
    # Required by IFilter
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # For this filter, we assume it's the primary colorization step, so it works on 'filtered_image'.
        return self.P(filtered_image, render_factor=render_factor, post_process=post_process)


    def process_sequence(self, frame_sequence: List[PilImage], render_factor: int = None, post_process: bool = True) -> PilImage:
        if len(frame_sequence) != self.n_frames_input:
            raise ValueError(f"Expected {self.n_frames_input} frames, got {len(frame_sequence)}")

        current_render_factor = render_factor if render_factor is not None else self.render_factor
        
        processed_frames_l_tensors = [] 
        middle_frame_idx = self.n_frames_input // 2
        orig_middle_frame_pil = frame_sequence[middle_frame_idx]
        
        target_w_render, target_h_render = orig_middle_frame_pil.size # Default if no render_factor
        if current_render_factor is not None:
            target_w_render, target_h_render = self.get_render_factor_target_size(orig_middle_frame_pil, current_render_factor)
        else:
            # Ensure divisible by 16 if no render factor
            target_w_render = target_w_render - target_w_render % 16
            target_h_render = target_h_render - target_h_render % 16
            if target_w_render == 0 or target_h_render == 0:
                 target_w_render, target_h_render = orig_middle_frame_pil.size


        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_cache():
            ab_output_tensor = self.model(input_tensor).squeeze(0) 

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Resize ab_output_tensor to match original middle frame's size
        # Use FastAIImage for its resize method that works on tensors directly
        # Note: fastai.vision.image.Image is typically aliased as `Image` in fastai
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
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


# Imports for VideoColorizerFilter
import torch
from torch import Tensor # Already imported via fastai.torch_core
import torch.nn as nn # Already imported via fastai.vision
# PilImage is already imported as from PIL import Image as PilImage
# from fastai.vision.image import Image is already imported via from fastai.vision.image import *
# from fastai.vision import transform as vision_transform # transform is available via from fastai.vision import *
# from typing import List, Tuple # Already imported via fastai.core -> from .imports.core import *
from .augs import denorm_lab_mean_std_tensor
# IFilter is defined in this file


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
    
    # Required by IFilter
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # 'filtered_image' is the image to be processed by this filter stage.
        # 'orig_image' is the original image from the very start of the pipeline.
        # For this filter, we assume it's the primary colorization step, so it works on 'filtered_image'.
        return self.P(filtered_image, render_factor=render_factor, post_process=post_process)


    def process_sequence(self, frame_sequence: List[PilImage], render_factor: int = None, post_process: bool = True) -> PilImage:
        if len(frame_sequence) != self.n_frames_input:
            raise ValueError(f"Expected {self.n_frames_input} frames, got {len(frame_sequence)}")

        current_render_factor = render_factor if render_factor is not None else self.render_factor
        
        processed_frames_l_tensors = [] 
        middle_frame_idx = self.n_frames_input // 2
        orig_middle_frame_pil = frame_sequence[middle_frame_idx]
        
        target_w_render, target_h_render = orig_middle_frame_pil.size # Default if no render_factor
        if current_render_factor is not None:
            target_w_render, target_h_render = self.get_render_factor_target_size(orig_middle_frame_pil, current_render_factor)
        else:
            # Ensure divisible by 16 if no render factor
            target_w_render = target_w_render - target_w_render % 16
            target_h_render = target_h_render - target_h_render % 16
            if target_w_render == 0 or target_h_render == 0:
                 target_w_render, target_h_render = orig_middle_frame_pil.size


        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            frame_tensor_l = vision.transform.ToTensor()(frame_pil_l_resized) 
            processed_frames_l_tensors.append(frame_tensor_l)
        
        input_tensor = torch.cat(processed_frames_l_tensors, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        with torch.no_cache():
            ab_output_tensor = self.model(input_tensor).squeeze(0) 

        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Resize ab_output_tensor to match original middle frame's size
        # Use FastAIImage for its resize method that works on tensors directly
        # Note: fastai.vision.image.Image is typically aliased as `Image` in fastai
        ab_output_resized = Image(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size).data
        
        middle_frame_l_tensor = vision.transform.ToTensor()(middle_frame_orig_l_pil) 
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, ab_output_resized)
        if post_process:
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision.transform.ToPILImage()(full_img_tensor.cpu())
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


# Imports for VideoColorizerFilter
import torch
from torch import Tensor
import torch.nn as nn
# PILImage is already imported as PilImage
# from PIL.Image import Image as PILImage 
from fastai.vision.image import Image as FastAIImage # Renamed to avoid clash with PIL.Image if that was also imported directly
from fastai.vision import transform as vision_transform
from typing import List, Tuple # Already imported via fastai.core -> from .imports.core import *
from .augs import denorm_lab_mean_std_tensor
# IFilter is defined in this file


class VideoColorizerFilter(IFilter):
    def __init__(self, model: nn.Module, render_factor: int = None, n_frames_input: int = 5, **kwargs):
        super().__init__(**kwargs) # Pass any IFilter specific kwargs
        self.model = model
        self.render_factor = render_factor
        self.n_frames_input = n_frames_input
        # Get device from model parameters, assuming model is already on a device
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            # Handle case where model has no parameters (unlikely for real models)
            self.device = torch.device("cpu") # Default to CPU
            logging.warning("Warning: Model given to VideoColorizerFilter has no parameters. Defaulting device to CPU.")


    def P(self, pil_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        # This method is for single images, which VideoColorizerFilter might not primarily support
        # Or it could process a single image by duplicating it n_frames_input times
        # For now, let's make it clear it's not the primary path.
        # Option 1: Raise error
        # raise NotImplementedError("VideoColorizerFilter is designed for sequences of frames. Use process_sequence.")
        # Option 2: Implement by duplicating frame (simple, but maybe not desired behavior)
        # Create a sequence by repeating the frame
        frame_sequence = [pil_image] * self.n_frames_input
        return self.process_sequence(frame_sequence, render_factor, post_process)
    
    # Required by IFilter
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        # For IFilter compatibility, we'll use the P method which processes a single image by duplicating it.
        # 'filtered_image' is the image to be processed.
        return self.P(filtered_image, render_factor=render_factor, post_process=post_process)


    def process_sequence(self, frame_sequence: List[PilImage], render_factor: int = None, post_process: bool = True) -> PilImage:
        if len(frame_sequence) != self.n_frames_input:
            raise ValueError(f"Expected {self.n_frames_input} frames, got {len(frame_sequence)}")

        current_render_factor = render_factor if render_factor is not None else self.render_factor
        
        processed_frames_l = [] # Stores L-channel tensors
        # Target size for AB output should be based on the middle frame's original size after render_factor
        middle_frame_idx = self.n_frames_input // 2
        orig_middle_frame_pil = frame_sequence[middle_frame_idx]
        
        # Determine target H, W for model input based on render_factor applied to middle frame
        # All frames in sequence will be resized to this H_r, W_r before going to model
        if current_render_factor is not None:
            target_w_render, target_h_render = self.get_render_factor_target_size(orig_middle_frame_pil, current_render_factor)
        else:
            # If no render_factor, use original size but ensure it's divisible by 16 (or model's requirement)
            temp_w, temp_h = orig_middle_frame_pil.size
            target_w_render, target_h_render = temp_w - temp_w % 16, temp_h - temp_h % 16
            if target_w_render == 0 or target_h_render == 0: # handle very small images
                target_w_render, target_h_render = orig_middle_frame_pil.size


        for frame_pil in frame_sequence:
            frame_pil_l = frame_pil.convert('L')
            # Resize all frames in the sequence to the same target_w_render, target_h_render
            frame_pil_l_resized = frame_pil_l.resize((target_w_render, target_h_render), PilImage.BILINEAR)
            
            frame_tensor_l = vision_transform.ToTensor()(frame_pil_l_resized) # (1, H_r, W_r)
            processed_frames_l.append(frame_tensor_l)
        
        # Stack frames for the model input
        # AdvancedVideoModel placeholder expects (1, n_frames_input * 1, H_r, W_r)
        input_tensor = torch.cat(processed_frames_l, dim=0).unsqueeze(0) 
        input_tensor = input_tensor.to(self.device)

        # Model inference
        with torch.no_cache():
            ab_output_tensor = self.model(input_tensor).squeeze(0) # Expected (2, H_r, W_r)

        # Post-processing
        # Get original L channel of the middle frame, ensure it's L mode
        middle_frame_orig_l_pil = orig_middle_frame_pil.convert('L')
        
        # Resize ab_output_tensor to match original middle frame's size (before render_factor)
        # The AB channels should correspond to the original full resolution of the middle frame.
        color_img_ab = FastAIImage(ab_output_tensor.cpu()).resize(orig_middle_frame_pil.size) # FastAI Image for resize
        
        middle_frame_l_tensor = vision_transform.ToTensor()(middle_frame_orig_l_pil) # (1, H_orig, W_orig)
        
        full_img_tensor = self.merge_lab_tensors(middle_frame_l_tensor, color_img_ab.data)
        if post_process:
            # denorm_lab_mean_std_tensor expects specific mean/std, ensure they are available
            # or that the model output doesn't need this if not using ColorizerFilter's normalization
            # For now, assume it's needed and available.
            full_img_tensor = denorm_lab_mean_std_tensor(full_img_tensor) 
        
        result_img_pil = vision_transform.ToPILImage()(full_img_tensor.cpu())
        return result_img_pil

    def get_render_factor_target_size(self, pil_image: PilImage, render_factor: int) -> Tuple[int, int]:
        # This is identical to ColorizerFilter's version.
        # Could be a utility function if shared by many filters.
        if render_factor is None: return pil_image.size
        new_w = int(pil_image.width * render_factor / 100)
        new_h = int(pil_image.height * render_factor / 100)
        # Ensure dimensions are multiples of a number if model requires (e.g., 16)
        new_w = new_w - new_w % 16
        new_h = new_h - new_h % 16
        return max(16, new_w), max(16, new_h) # Ensure at least 16x16

    def merge_lab_tensors(self, t_l: Tensor, t_ab: Tensor) -> Tensor:
        # t_l is (1,H,W) from ToTensor, range [0,1]
        # t_ab is (2,H,W) from model, range e.g. [-1,1] or normalized
        # Standard LAB: L [0,100], A [-128,127], B [-128,127]
        # ColorizerFilter scales L by 100. If denorm_lab_mean_std_tensor is used,
        # it expects specific normalized ranges.
        # For now, follow ColorizerFilter:
        t_l_scaled = t_l * 100.0 
        return torch.cat([t_l_scaled, t_ab], dim=0) # (3, H, W)
