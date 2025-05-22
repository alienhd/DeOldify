from fastai.core import *
from fastai.vision import *
import numpy as np
from matplotlib.axes import Axes
from .filters import IFilter, MasterFilter, ColorizerFilter
from .generators import gen_inference_deep, gen_inference_wide
from PIL import Image
import ffmpeg
import yt_dlp as youtube_dl
import gc
import requests
from io import BytesIO
import base64
from IPython import display as ipythondisplay
from IPython.display import HTML
from IPython.display import Image as ipythonimage
import cv2
import logging
import shutil
import re


# adapted from https://www.pyimagesearch.com/2016/04/25/watermarking-images-with-opencv-and-python/
def get_watermarked(pil_image: Image) -> Image:
    try:
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        (h, w) = image.shape[:2]
        image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])
        pct = 0.05
        full_watermark = cv2.imread(
            './resource_images/watermark.png', cv2.IMREAD_UNCHANGED
        )
        (fwH, fwW) = full_watermark.shape[:2]
        wH = int(pct * h)
        wW = int((pct * h / fwH) * fwW)
        watermark = cv2.resize(full_watermark, (wH, wW), interpolation=cv2.INTER_AREA)
        overlay = np.zeros((h, w, 4), dtype="uint8")
        (wH, wW) = watermark.shape[:2]
        overlay[h - wH - 10 : h - 10, 10 : 10 + wW] = watermark
        # blend the two images together using transparent overlays
        output = image.copy()
        cv2.addWeighted(overlay, 0.5, output, 1.0, 0, output)
        rgb_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        final_image = Image.fromarray(rgb_image)
        return final_image
    except:
        # Don't want this to crash everything, so let's just not watermark the image for now.
        return pil_image


class ModelImageVisualizer:
    def __init__(self, filter: IFilter, results_dir: str = None):
        self.filter = filter
        self.results_dir = None if results_dir is None else Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _clean_mem(self):
        torch.cuda.empty_cache()
        # gc.collect()

    def _open_pil_image(self, path: Path) -> Image:
        return PIL.Image.open(path).convert('RGB')

    def _get_image_from_url(self, url: str) -> Image:
        response = requests.get(url, timeout=30, headers={'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36'})
        img = PIL.Image.open(BytesIO(response.content)).convert('RGB')
        return img

    def plot_transformed_image_from_url(
        self,
        url: str,
        path: str = 'test_images/image.png',
        results_dir:Path = None,
        figsize: Tuple[int, int] = (20, 20),
        render_factor: int = None,
        
        display_render_factor: bool = False,
        compare: bool = False,
        post_process: bool = True,
        watermarked: bool = True,
    ) -> Path:
        img = self._get_image_from_url(url)
        img.save(path)
        return self.plot_transformed_image(
            path=path,
            results_dir=results_dir,
            figsize=figsize,
            render_factor=render_factor,
            display_render_factor=display_render_factor,
            compare=compare,
            post_process = post_process,
            watermarked=watermarked,
        )

    def plot_transformed_image(
        self,
        path: str,
        results_dir:Path = None,
        figsize: Tuple[int, int] = (20, 20),
        render_factor: int = None,
        display_render_factor: bool = False,
        compare: bool = False,
        post_process: bool = True,
        watermarked: bool = True,
    ) -> Path:
        path = Path(path)
        if results_dir is None:
            results_dir = Path(self.results_dir)
        result = self.get_transformed_image(
            path, render_factor, post_process=post_process,watermarked=watermarked
        )
        orig = self._open_pil_image(path)
        if compare:
            self._plot_comparison(
                figsize, render_factor, display_render_factor, orig, result
            )
        else:
            self._plot_solo(figsize, render_factor, display_render_factor, result)

        orig.close()
        result_path = self._save_result_image(path, result, results_dir=results_dir)
        result.close()
        return result_path

    def _plot_comparison(
        self,
        figsize: Tuple[int, int],
        render_factor: int,
        display_render_factor: bool,
        orig: Image,
        result: Image,
    ):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        self._plot_image(
            orig,
            axes=axes[0],
            figsize=figsize,
            render_factor=render_factor,
            display_render_factor=False,
        )
        self._plot_image(
            result,
            axes=axes[1],
            figsize=figsize,
            render_factor=render_factor,
            display_render_factor=display_render_factor,
        )

    def _plot_solo(
        self,
        figsize: Tuple[int, int],
        render_factor: int,
        display_render_factor: bool,
        result: Image,
    ):
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        self._plot_image(
            result,
            axes=axes,
            figsize=figsize,
            render_factor=render_factor,
            display_render_factor=display_render_factor,
        )

    def _save_result_image(self, source_path: Path, image: Image, results_dir = None) -> Path:
        if results_dir is None:
            results_dir = Path(self.results_dir)
        result_path = results_dir / source_path.name
        image.save(result_path)
        return result_path

    def get_transformed_image(
        self, path: Path, render_factor: int = None, post_process: bool = True,
        watermarked: bool = True,
    ) -> Image:
        self._clean_mem()
        orig_image = self._open_pil_image(path)
        filtered_image = self.filter.filter(
            orig_image, orig_image, render_factor=render_factor,post_process=post_process
        )

        if watermarked:
            return get_watermarked(filtered_image)

        return filtered_image

    def _plot_image(
        self,
        image: Image,
        render_factor: int,
        axes: Axes = None,
        figsize=(20, 20),
        display_render_factor = False,
    ):
        if axes is None:
            _, axes = plt.subplots(figsize=figsize)
        axes.imshow(np.asarray(image) / 255)
        axes.axis('off')
        if render_factor is not None and display_render_factor:
            plt.text(
                10,
                10,
                'render_factor: ' + str(render_factor),
                color='white',
                backgroundcolor='black',
            )

    def _get_num_rows_columns(self, num_images: int, max_columns: int) -> Tuple[int, int]:
        columns = min(num_images, max_columns)
        rows = num_images // columns
        rows = rows if rows * columns == num_images else rows + 1
        return rows, columns


class VideoColorizer:
    def __init__(self, vis: ModelImageVisualizer):
        self.vis = vis
        workfolder = Path('./video')
        self.source_folder = workfolder / "source"
        self.bwframes_root = workfolder / "bwframes"
        self.audio_root = workfolder / "audio"
        self.colorframes_root = workfolder / "colorframes"
        self.result_folder = workfolder / "result"
        self.video_processor = VideoProcessor(
            source_folder=self.source_folder,
            bwframes_root=self.bwframes_root,
            audio_root=self.audio_root,
            colorframes_root=self.colorframes_root,
            result_folder=self.result_folder,
        )

    def _colorize_raw_frames(
        self, source_path: Path, render_factor: int = None, post_process: bool = True,
        watermarked: bool = True,
    ):
        colorframes_folder = self.colorframes_root / (source_path.stem)
        colorframes_folder.mkdir(parents=True, exist_ok=True)
        self.video_processor._purge_images(colorframes_folder)
        bwframes_folder = self.bwframes_root / (source_path.stem)

        for img in progress_bar(os.listdir(str(bwframes_folder))):
            img_path = bwframes_folder / img

            if os.path.isfile(str(img_path)):
                color_image = self.vis.get_transformed_image(
                    str(img_path), render_factor=render_factor, post_process=post_process,watermarked=watermarked
                )
                color_image.save(str(colorframes_folder / img))

    def colorize_from_url(
        self,
        source_url,
        base_file_name: str, # Changed from file_name, expected to be extensionless
        render_factor: int = None,
        post_process: bool = True,
        watermarked: bool = True,

    ) -> Path:
        # base_file_name is expected to be without extension, e.g., "my_video"
        base_source_path = self.video_processor.source_folder / base_file_name
        actual_source_path = self.video_processor._download_video_from_url(source_url, base_source_path)
        return self._colorize_from_path(
            actual_source_path, render_factor=render_factor, post_process=post_process,watermarked=watermarked
        )

    def colorize_from_file_name(
        self, file_name: str, render_factor: int = None,  watermarked: bool = True, post_process: bool = True,
    ) -> Path:
        source_path = self.source_folder / file_name
        return self._colorize_from_path(
            source_path, render_factor=render_factor,  post_process=post_process,watermarked=watermarked
        )

    def _colorize_from_path(
        self, source_path: Path, render_factor: int = None,  watermarked: bool = True, post_process: bool = True
    ) -> Path:
        if not source_path.exists():
            raise Exception(
                'Video at path specfied, ' + str(source_path) + ' could not be found.'
            )
        self.video_processor._extract_raw_frames(source_path)
        self._colorize_raw_frames(
            source_path, render_factor=render_factor,post_process=post_process,watermarked=watermarked
        )
        self._apply_temporal_smoothing(source_path)
        return self.video_processor._build_video(source_path)

    def _apply_temporal_smoothing(self, source_path: Path, alpha: float = 0.7):
        colorframes_folder = self.colorframes_root / (source_path.stem)
        if not colorframes_folder.exists():
            logging.warning(f"Colorized frames folder not found: {colorframes_folder}. Skipping temporal smoothing.")
            return

        frame_files = sorted([f for f in os.listdir(str(colorframes_folder)) if re.search(r'.*?\.jpg$', f)])

        if len(frame_files) < 3:
            logging.info("Not enough frames for temporal smoothing. Skipping.")
            return

        logging.info(f"Applying temporal smoothing to {len(frame_files)} frames for {source_path.name}...")

        img_prev = None
        img_curr = None
        img_next = None
        img_smoothed_curr = None # Stores the result of smoothing img_curr
        blended_neighbors = None 

        try:
            # Initialize by loading the first two images
            img_prev_path = colorframes_folder / frame_files[0]
            img_prev = self.vis._open_pil_image(img_prev_path)
            
            img_curr_path = colorframes_folder / frame_files[1]
            img_curr = self.vis._open_pil_image(img_curr_path)
        except Exception as e:
            logging.error(f"Could not open initial frames for temporal smoothing in {colorframes_folder}. Error: {e}", exc_info=True)
            if img_prev and hasattr(img_prev, 'close'): img_prev.close()
            if img_curr and hasattr(img_curr, 'close'): img_curr.close()
            return

        logging.info(f"Starting temporal smoothing loop for frames in {colorframes_folder}...")
        # Loop from the second frame (index 1) up to the second-to-last frame (index len(frame_files) - 2)
        # The frame being modified and saved is img_files[i], which corresponds to the initial img_curr
        for i in range(1, len(frame_files) - 1):
            img_next_path = None # For logging in except block
            try:
                img_next_path = colorframes_folder / frame_files[i+1]
                if img_next: img_next.close() # Close previous img_next
                img_next = self.vis._open_pil_image(img_next_path)

                # 1. Blend neighbors: blended_neighbors = 0.5*img_prev + 0.5*img_next
                if blended_neighbors: blended_neighbors.close()
                blended_neighbors = Image.blend(img_prev, img_next, 0.5)
                
                # 2. Blend current with blended_neighbors: result = alpha*img_curr + (1-alpha)*blended_neighbors
                # Image.blend(B, A, alpha_curr_weight) means (1-alpha_curr_weight)*B + alpha_curr_weight*A
                # Here A is img_curr, B is blended_neighbors. alpha is weight of img_curr.
                if img_smoothed_curr: img_smoothed_curr.close()
                img_smoothed_curr = Image.blend(blended_neighbors, img_curr, alpha)

                output_frame_path = colorframes_folder / frame_files[i] # This is path for original img_curr
                logging.debug(f"Saving temporally smoothed frame: {output_frame_path.name}")
                img_smoothed_curr.save(output_frame_path)

                # Shift frames for the next iteration: current becomes previous, next becomes current
                if img_prev: img_prev.close() 
                img_prev = img_curr
                img_curr = img_next
                img_next = None # Will be loaded at the start of the next iteration

            except Exception as e:
                logging.error(f"Error during temporal smoothing for frame {frame_files[i]} (path: {colorframes_folder/frame_files[i]}). Next frame was {img_next_path}. Details: {e}", exc_info=True)
                # Attempt to close images if they are open
                if img_next is not None and hasattr(img_next, 'close'): img_next.close()
                # Let the outer finally block handle img_prev, img_curr, img_smoothed_curr, blended_neighbors
                break # Exit loop on error
            finally:
                # Close intermediate images created in this iteration
                if img_smoothed_curr and hasattr(img_smoothed_curr, 'close'):
                    try:
                        if hasattr(img_smoothed_curr, 'fp') and img_smoothed_curr.fp and not img_smoothed_curr.fp.closed:
                            img_smoothed_curr.close()
                    except Exception: pass
                    img_smoothed_curr = None 
                if blended_neighbors and hasattr(blended_neighbors, 'close'):
                    try:
                        if hasattr(blended_neighbors, 'fp') and blended_neighbors.fp and not blended_neighbors.fp.closed:
                             blended_neighbors.close()
                    except Exception: pass
                    blended_neighbors = None

        # Close the last two images held in img_prev and img_curr after the loop finishes
        if img_prev and hasattr(img_prev, 'close'):
            try:
                if hasattr(img_prev, 'fp') and img_prev.fp and not img_prev.fp.closed: img_prev.close()
            except Exception: pass
        if img_curr and hasattr(img_curr, 'close'):
            try:
                if hasattr(img_curr, 'fp') and img_curr.fp and not img_curr.fp.closed: img_curr.close()
            except Exception: pass
        # img_next should be None here if loop completed, or closed in case of error.
        # img_smoothed_curr and blended_neighbors are closed in the loop's finally or here if loop didn't run.
        if img_smoothed_curr and hasattr(img_smoothed_curr, 'close'):
            try:
                if hasattr(img_smoothed_curr, 'fp') and img_smoothed_curr.fp and not img_smoothed_curr.fp.closed: img_smoothed_curr.close()
            except Exception: pass
        if blended_neighbors and hasattr(blended_neighbors, 'close'):
            try:
                if hasattr(blended_neighbors, 'fp') and blended_neighbors.fp and not blended_neighbors.fp.closed: blended_neighbors.close()
            except Exception: pass


        logging.info(f"Temporal smoothing completed for frames in {colorframes_folder}.")


class VideoProcessor:
    def __init__(self, source_folder: Path, bwframes_root: Path, audio_root: Path, colorframes_root: Path, result_folder: Path):
        self.source_folder = source_folder
        self.bwframes_root = bwframes_root
        self.audio_root = audio_root
        self.colorframes_root = colorframes_root
        self.result_folder = result_folder

    def _get_ffmpeg_probe(self, path:Path):
        try:
            probe = ffmpeg.probe(str(path))
            return probe
        except ffmpeg.Error as e:
            # Attempt to reconstruct the conceptual "command" for logging
            # For ffmpeg.probe, the arguments are part of the function call.
            # We can log the path being probed.
            cmd_str = f"ffmpeg.probe(filename='{str(path)}')"
            error_message = (
                f"ffmpeg failed during video probing for {path.name}.\n"
                f"Attempted operation: {cmd_str}\n"
                f"FFmpeg stdout: {e.stdout.decode('utf-8') if e.stdout else 'N/A'}\n"
                f"FFmpeg stderr: {e.stderr.decode('utf-8') if e.stderr else 'N/A'}"
            )
            logging.error(error_message, exc_info=True)
            raise e
        except Exception as e:
            logging.error(f"An unexpected error occurred during ffmpeg.probe for {path.name}.", exc_info=True)
            raise e

    def _purge_images(self, dir):
        for f in os.listdir(dir):
            if re.search('.*?\.jpg', f):
                os.remove(os.path.join(dir, f))

    def _get_fps(self, source_path: Path) -> str:
        probe = self._get_ffmpeg_probe(source_path)
        stream_data = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
            None,
        )
        return stream_data['avg_frame_rate']

    def _download_video_from_url(self, source_url: str, base_source_path: Path) -> Path:
        # base_source_path is expected to be like /video/source/my_video_title (no extension)
        # Clean up any existing files that might match the base name + any extension
        existing_files = list(base_source_path.parent.glob(f"{base_source_path.name}.*"))
        for f_exist in existing_files:
            logging.info(f"Removing existing file {f_exist} before download.")
            f_exist.unlink()

        ydl_opts = {
            'format': 'bestvideo+bestaudio/best', # Download best available format
            'outtmpl': str(base_source_path) + '.%(ext)s', # yt-dlp will add the correct extension
            'retries': 30,
            'fragment-retries': 30,
            'quiet': True, # Suppress yt-dlp console output unless errors
            'merge_output_format': 'mp4' # if merging is needed, prefer mp4, but format selection above is primary
        }
        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([source_url])
        except yt_dlp.utils.DownloadError as e:
            logging.error(f"Failed to download video from URL: {source_url}. yt_dlp DownloadError: {e}", exc_info=True)
            raise e
        except Exception as e:
            logging.error(f"An unexpected error occurred while trying to download video from URL: {source_url}. Error: {e}", exc_info=True)
            raise e
        
        # Find the actual downloaded file
        # yt-dlp should have added an extension based on '.%(ext)s'
        downloaded_files = sorted(base_source_path.parent.glob(f"{base_source_path.name}.*"), key=os.path.getmtime, reverse=True)
        
        if not downloaded_files:
            raise Exception(f"Failed to find downloaded file for base {base_source_path.name} in {base_source_path.parent}")
        
        actual_source_path = downloaded_files[0]
        logging.info(f"Video downloaded to: {actual_source_path}")
        return actual_source_path

    def _extract_raw_frames(self, source_path: Path):
        bwframes_folder = self.bwframes_root / (source_path.stem)
        bwframe_path_template = str(bwframes_folder / '%5d.jpg')
        bwframes_folder.mkdir(parents=True, exist_ok=True)
        self._purge_images(bwframes_folder)

        process = (
            ffmpeg
                .input(str(source_path))
                .output(str(bwframe_path_template), format='image2', vcodec='mjpeg', **{'q:v':'0'})
                .global_args('-hide_banner')
                .global_args('-nostats')
                .global_args('-loglevel', 'error')
        )

        try:
            # The process.run() method in ffmpeg-python returns (stdout, stderr)
            # and raises ffmpeg.Error on non-zero return code.
            # Ensure capture_stdout and capture_stderr are True if you need to access them outside the exception.
            # However, e.stdout and e.stderr are populated in the exception object.
            process.run(capture_stdout=True, capture_stderr=True) 
        except ffmpeg.Error as e:
            error_message = (
                f"ffmpeg failed during raw frame extraction for {source_path.name}.\n"
                f"Command: {' '.join(process.args)}\n"
                f"FFmpeg stdout: {e.stdout.decode('utf-8') if e.stdout else 'N/A'}\n"
                f"FFmpeg stderr: {e.stderr.decode('utf-8') if e.stderr else 'N/A'}"
            )
            logging.error(error_message, exc_info=True)
            raise e
        except Exception as e:
            logging.error(f"An unexpected error occurred during raw frame extraction for {source_path.name}.", exc_info=True)
            raise e

    def _build_video(self, source_path: Path) -> Path:
        # Output paths should consistently use .mp4 extension
        colorized_path = self.result_folder / (source_path.stem + '_no_audio.mp4')
        result_path = self.result_folder / (source_path.stem + '.mp4')

        colorframes_folder = self.colorframes_root / (source_path.stem)
        colorframes_path_template = str(colorframes_folder / '%5d.jpg')
        
        colorized_path.parent.mkdir(parents=True, exist_ok=True)
        if colorized_path.exists():
            colorized_path.unlink()
        
        fps = self._get_fps(source_path)

        process = (
            ffmpeg
                .input(str(colorframes_path_template), format='image2', vcodec='mjpeg', framerate=fps)
                .output(str(colorized_path), crf=17, vcodec='libx264') # Ensures MP4 output for colorized part
                .global_args('-hide_banner')
                .global_args('-nostats')
                .global_args('-loglevel', 'error')
        )

        try:
            process.run(capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            error_message = (
                f"ffmpeg failed during video building (pass 1 - creating video from colorized frames) for {source_path.name}.\n"
                f"Command: {' '.join(process.args)}\n"
                f"FFmpeg stdout: {e.stdout.decode('utf-8') if e.stdout else 'N/A'}\n"
                f"FFmpeg stderr: {e.stderr.decode('utf-8') if e.stderr else 'N/A'}"
            )
            logging.error(error_message, exc_info=True)
            raise e
        except Exception as e:
            logging.error(f"An unexpected error occurred during video building (pass 1) for {source_path.name}.", exc_info=True)
            raise e

        if result_path.exists():
            result_path.unlink()
        # making copy of non-audio version in case adding back audio doesn't apply or fails.
        shutil.copyfile(str(colorized_path), str(result_path))

        # adding back sound here
        # Audio file should be uniquely named based on source stem and stored in audio_root
        self.audio_root.mkdir(parents=True, exist_ok=True) # Ensure audio_root exists
        audio_file = self.audio_root / (source_path.stem + '.aac')
        
        if audio_file.exists():
            audio_file.unlink()

        # Audio extraction from the original source_path (could be .mkv, .webm, etc.)
        audio_extract_cmd = [
            'ffmpeg', '-y', '-i', str(source_path),
            '-vn', '-acodec', 'copy', str(audio_file),
            '-hide_banner', '-nostats', '-loglevel', 'error'
        ]
        try:
            # Using subprocess for better error handling potential, though os.system is kept for now.
            # For os.system, we can't easily get stderr/stdout or specific exceptions.
            # A non-zero return code indicates failure but not details.
            ret_code = os.system(' '.join(audio_extract_cmd))
            if ret_code != 0:
                logging.error(f"Audio extraction command failed with exit code {ret_code} for {source_path.name}. Command: {' '.join(audio_extract_cmd)}", exc_info=False) # exc_info=False as there's no exception object here
                # Not raising an error here to match previous behavior of os.system, but logging it.
        except Exception as e:
            logging.error(f"An unexpected error occurred during audio extraction for {source_path.name}. Command: {' '.join(audio_extract_cmd)}", exc_info=True)
            # Not raising error to keep flow similar to plain os.system if it were to somehow raise

        if audio_file.exists():
            audio_merge_cmd = [
                'ffmpeg', '-y', '-i', str(colorized_path), '-i', str(audio_file),
                '-shortest', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k', str(result_path),
                '-hide_banner', '-nostats', '-loglevel', 'error'
            ]
            try:
                ret_code = os.system(' '.join(audio_merge_cmd))
                if ret_code != 0:
                    logging.error(f"Audio merge command failed with exit code {ret_code} for {source_path.name}. Command: {' '.join(audio_merge_cmd)}", exc_info=False)
            except Exception as e:
                logging.error(f"An unexpected error occurred during audio merging for {source_path.name}. Command: {' '.join(audio_merge_cmd)}", exc_info=True)

        logging.info('Video created here: ' + str(result_path))
        return result_path


def get_video_colorizer(render_factor: int = 21) -> VideoColorizer:
    return get_stable_video_colorizer(render_factor=render_factor)


def get_artistic_video_colorizer(
    root_folder: Path = Path('./'),
    weights_name: str = 'ColorizeArtistic_gen',
    results_dir='result_images',
    render_factor: int = 35
) -> VideoColorizer:
    learn = gen_inference_deep(root_folder=root_folder, weights_name=weights_name)
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filtr, results_dir=results_dir)
    return VideoColorizer(vis)


def get_stable_video_colorizer(
    root_folder: Path = Path('./'),
    weights_name: str = 'ColorizeVideo_gen',
    results_dir='result_images',
    render_factor: int = 21
) -> VideoColorizer:
    learn = gen_inference_wide(root_folder=root_folder, weights_name=weights_name)
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filtr, results_dir=results_dir)
    return VideoColorizer(vis)


def get_image_colorizer(
    root_folder: Path = Path('./'), render_factor: int = 35, artistic: bool = True
) -> ModelImageVisualizer:
    if artistic:
        return get_artistic_image_colorizer(root_folder=root_folder, render_factor=render_factor)
    else:
        return get_stable_image_colorizer(root_folder=root_folder, render_factor=render_factor)


def get_stable_image_colorizer(
    root_folder: Path = Path('./'),
    weights_name: str = 'ColorizeStable_gen',
    results_dir='result_images',
    render_factor: int = 35
) -> ModelImageVisualizer:
    learn = gen_inference_wide(root_folder=root_folder, weights_name=weights_name)
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filtr, results_dir=results_dir)
    return vis


def get_artistic_image_colorizer(
    root_folder: Path = Path('./'),
    weights_name: str = 'ColorizeArtistic_gen',
    results_dir='result_images',
    render_factor: int = 35
) -> ModelImageVisualizer:
    learn = gen_inference_deep(root_folder=root_folder, weights_name=weights_name)
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filtr, results_dir=results_dir)
    return vis


def show_image_in_notebook(image_path: Path):
    ipythondisplay.display(ipythonimage(str(image_path)))


def show_video_in_notebook(video_path: Path):
    video = io.open(video_path, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(
        HTML(
            data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(
                encoded.decode('ascii')
            )
        )
    )
