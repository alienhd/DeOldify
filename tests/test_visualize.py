import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import os
import sys
from pathlib import Path
import re # For re.search in test_purge_images
import yt_dlp # Import for yt_dlp.utils.DownloadError
import shutil # For shutil.copyfile
import ffmpeg # For ffmpeg.Error

# Add the project root to sys.path to allow importing deoldify
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Mock torch before importing deoldify components that might use it for device selection
# This is crucial to prevent errors in CPU-only test environments
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False # Simulate no GPU
mock_torch.device.return_value = 'cpu' # Simulate CPU device
mock_torch.cuda.set_device = MagicMock() # Mock set_device
mock_torch.cuda.empty_cache = MagicMock() # Mock empty_cache
sys.modules['torch'] = mock_torch

# Mock parts of deoldify that might do device setup at import time
with patch('deoldify._device._Device') as MockDeviceSingleton: 
    from deoldify.visualize import VideoProcessor, VideoColorizer
    from deoldify.filters import IFilter 


class TestVideoProcessor(unittest.TestCase):
    def setUp(self):
        # Common dummy paths for VideoProcessor instantiation
        self.dummy_source_folder = Path('dummy_video/source')
        self.dummy_bwframes_root = Path('dummy_video/bwframes')
        self.dummy_audio_root = Path('dummy_video/audio')
        self.dummy_colorframes_root = Path('dummy_video/colorframes')
        self.dummy_result_folder = Path('dummy_video/result')

        # Base VideoProcessor instance for tests that don't need specific path mocks for constructor
        self.vp = VideoProcessor(
            source_folder=self.dummy_source_folder,
            bwframes_root=self.dummy_bwframes_root,
            audio_root=self.dummy_audio_root,
            colorframes_root=self.dummy_colorframes_root,
            result_folder=self.dummy_result_folder
        )

    def test_video_processor_initialization(self):
        # Test with paths from setUp
        self.assertEqual(self.vp.source_folder, self.dummy_source_folder)
        self.assertEqual(self.vp.bwframes_root, self.dummy_bwframes_root)
        self.assertEqual(self.vp.audio_root, self.dummy_audio_root)
        self.assertEqual(self.vp.colorframes_root, self.dummy_colorframes_root)
        self.assertEqual(self.vp.result_folder, self.dummy_result_folder)

    @patch('os.remove')
    @patch('os.listdir')
    @patch('re.search') 
    def test_purge_images(self, mock_re_search, mock_os_listdir, mock_os_remove):
        mock_os_listdir.return_value = ['1.jpg', '2.png', '3.txt', '4.JPG']
        
        def search_side_effect(pattern, filename_str):
            if filename_str.lower().endswith('.jpg'): 
                return MagicMock() 
            return None
        mock_re_search.side_effect = search_side_effect
        
        test_dir_str = 'dummy_dir_scenario1' 
        self.vp._purge_images(test_dir_str) 

        expected_remove_calls = [
            call(os.path.join(test_dir_str, '1.jpg')),
            call(os.path.join(test_dir_str, '4.JPG')) 
        ]
        mock_os_remove.assert_has_calls(expected_remove_calls, any_order=True)
        self.assertEqual(mock_os_remove.call_count, 2)

        mock_os_listdir.return_value = ['2.png', '3.txt']
        mock_os_remove.reset_mock() 
        mock_re_search.side_effect = search_side_effect 
        
        test_dir_path_str_scenario2 = 'dummy_dir_scenario2'
        self.vp._purge_images(test_dir_path_str_scenario2)
        mock_os_remove.assert_not_called()

    @patch('ffmpeg.probe')
    def test_get_ffmpeg_probe_success(self, mock_ffmpeg_probe):
        dummy_path = Path('video.mp4')
        expected_probe_data = {'format': {'duration': '10.0'}}
        mock_ffmpeg_probe.return_value = expected_probe_data
        
        probe_data = self.vp._get_ffmpeg_probe(dummy_path)
        
        mock_ffmpeg_probe.assert_called_once_with(str(dummy_path))
        self.assertEqual(probe_data, expected_probe_data)

    @patch('ffmpeg.probe')
    @patch('logging.error') 
    def test_get_ffmpeg_probe_failure(self, mock_logging_error, mock_ffmpeg_probe):
        dummy_path = Path('non_existent_video.mp4')
        
        # Using ffmpeg.Error directly as it's an Exception subclass
        # The actual error in the code has stdout and stderr attributes.
        mock_ffmpeg_probe.side_effect = ffmpeg.Error("ffmpeg probe error", stdout=b"ffmpeg_stdout_output", stderr=b"ffmpeg_stderr_output: Probe failed")
        
        with self.assertRaises(ffmpeg.Error): 
            self.vp._get_ffmpeg_probe(dummy_path)
        
        self.assertTrue(mock_logging_error.called)
        args, kwargs = mock_logging_error.call_args_list[0] 
        logged_message = args[0]

        self.assertIn(f"ffmpeg failed during video probing for {dummy_path.name}", logged_message)
        self.assertIn("FFmpeg stdout: ffmpeg_stdout_output", logged_message)
        self.assertIn("FFmpeg stderr: ffmpeg_stderr_output: Probe failed", logged_message)

    @patch.object(VideoProcessor, '_get_ffmpeg_probe')
    def test_get_fps_success(self, mock_get_probe):
        dummy_path = Path('video.mp4')
        mock_get_probe.return_value = {
            'streams': [
                {'codec_type': 'video', 'avg_frame_rate': '30000/1001'},
                {'codec_type': 'audio', 'avg_frame_rate': '0/0'}
            ]
        }
        fps = self.vp._get_fps(dummy_path)
        mock_get_probe.assert_called_once_with(dummy_path)
        self.assertEqual(fps, '30000/1001')

    @patch.object(VideoProcessor, '_get_ffmpeg_probe')
    def test_get_fps_no_video_stream(self, mock_get_probe):
        dummy_path = Path('audio_only.mp4')
        mock_get_probe.return_value = {
            'streams': [{'codec_type': 'audio', 'avg_frame_rate': '0/0'}]
        }
        # The current code in visualize.py will try to access stream_data['avg_frame_rate']
        # where stream_data is None if no video stream is found. This causes a TypeError.
        with self.assertRaises(TypeError): 
             self.vp._get_fps(dummy_path)
        mock_get_probe.assert_called_once_with(dummy_path)


    @patch.object(VideoProcessor, '_get_ffmpeg_probe')
    def test_get_fps_probe_fails(self, mock_get_probe):
        dummy_path = Path('corrupt_video.mp4')
        mock_get_probe.side_effect = Exception("Probe failed") # Simulate probe failure
        with self.assertRaises(Exception) as context:
            self.vp._get_fps(dummy_path)
        self.assertIn("Probe failed", str(context.exception))

    @patch('yt_dlp.YoutubeDL') 
    @patch('pathlib.Path.unlink') 
    @patch('pathlib.Path.glob') 
    def test_download_video_from_url_success(self, mock_glob, mock_unlink, mock_YoutubeDL):
        mock_ydl_instance = MagicMock()
        mock_YoutubeDL.return_value.__enter__.return_value = mock_ydl_instance
       
        source_url = 'http://example.com/video.mp4'
        base_file_name_stem = 'test_video'
        # base_source_path is constructed using vp.source_folder
        base_source_path = self.vp.source_folder / base_file_name_stem

        downloaded_file_path = base_source_path.with_suffix('.mp4')
        
        # First glob call (for cleanup) returns no existing files
        # Second glob call (for finding downloaded file) returns the target file
        mock_glob.side_effect = [
            [], 
            [downloaded_file_path]
        ]
            
        actual_path = self.vp._download_video_from_url(source_url, base_source_path)
       
        # Check glob calls
        self.assertEqual(mock_glob.call_count, 2)
        # First call to glob (cleanup on base_source_path.parent)
        self.assertEqual(mock_glob.call_args_list[0], call(f"{base_source_path.name}.*"))
        # Second call to glob (finding downloaded file on base_source_path.parent)
        self.assertEqual(mock_glob.call_args_list[1], call(f"{base_source_path.name}.*"))
        
        mock_unlink.assert_not_called() # Because the first glob call returned empty

        mock_YoutubeDL.assert_called_once_with({
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': str(base_source_path) + '.%(ext)s',
            'retries': 30,
            'fragment-retries': 30,
            'quiet': True,
            'merge_output_format': 'mp4'
        })
        mock_ydl_instance.download.assert_called_once_with([source_url])
        self.assertEqual(actual_path, downloaded_file_path)

    @patch('yt_dlp.YoutubeDL')
    @patch('logging.error')
    @patch('pathlib.Path.glob') 
    def test_download_video_from_url_download_fails(self, mock_cleanup_glob, mock_logging_error, mock_YoutubeDL):
        mock_cleanup_glob.return_value = [] 

        mock_ydl_instance = MagicMock()
        mock_YoutubeDL.return_value.__enter__.return_value = mock_ydl_instance
        # Use the actual error class from yt_dlp.utils
        mock_ydl_instance.download.side_effect = yt_dlp.utils.DownloadError("Download error from test", Exception())
       
        source_url = 'http://example.com/non_existent_video.mp4'
        base_source_path = self.vp.source_folder / 'test_video_fail'
       
        with self.assertRaises(yt_dlp.utils.DownloadError) as context:
            self.vp._download_video_from_url(source_url, base_source_path)
       
        self.assertIn("Download error from test", str(context.exception))
        self.assertTrue(mock_logging_error.called)
        self.assertIn(f"Failed to download video from URL: {source_url}", mock_logging_error.call_args[0][0])

    @patch('ffmpeg.input') 
    @patch.object(VideoProcessor, '_purge_images') 
    @patch('pathlib.Path.mkdir') 
    def test_extract_raw_frames_success(self, mock_general_mkdir, mock_purge_images, mock_ffmpeg_input_module):
        mock_ffmpeg_run_method = MagicMock()
        mock_input_instance = MagicMock()
        mock_output_instance = MagicMock()
        mock_global_args_instance = MagicMock()

        mock_ffmpeg_input_module.return_value = mock_input_instance
        mock_input_instance.output.return_value = mock_output_instance
        mock_output_instance.global_args.return_value = mock_global_args_instance
        mock_global_args_instance.run = mock_ffmpeg_run_method
       
        source_path = Path('test_video.mp4')
        bwframes_folder = self.vp.bwframes_root / source_path.stem
       
        self.vp._extract_raw_frames(source_path)

        # Check if bwframes_folder.mkdir was called.
        mkdir_called_on_bwframes = False
        for c_args, c_kwargs in mock_general_mkdir.call_args_list:
            if len(c_args) > 0 and isinstance(c_args[0], Path) and str(c_args[0]) == str(bwframes_folder):
                 if c_kwargs == {'parents': True, 'exist_ok': True}:
                    mkdir_called_on_bwframes = True
                    break
        self.assertTrue(mkdir_called_on_bwframes, f"bwframes_folder.mkdir(parents=True, exist_ok=True) not called correctly on {bwframes_folder}")


        mock_purge_images.assert_called_once_with(bwframes_folder)
       
        expected_template = str(bwframes_folder / '%5d.jpg')
        mock_ffmpeg_input_module.assert_called_once_with(str(source_path))
        
        args, kwargs = mock_input_instance.output.call_args
        self.assertEqual(args[0], expected_template)
        self.assertEqual(kwargs['format'], 'image2')
        self.assertEqual(kwargs['vcodec'], 'mjpeg')
        self.assertEqual(kwargs['q:v'], '0') 

        mock_ffmpeg_run_method.assert_called_once_with(capture_stdout=True, capture_stderr=True)

    @patch('ffmpeg.input')
    @patch('os.system')
    @patch('shutil.copyfile')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.unlink')
    @patch.object(VideoProcessor, '_get_fps')
    # _apply_watermark_to_video was added in visualize.py (Turn 9), so it should be mocked here.
    @patch.object(VideoProcessor, '_apply_watermark_to_video') 
    @patch('pathlib.Path.mkdir') 
    def test_build_video_success_with_audio_no_watermark(self, mock_general_mkdir, mock_apply_watermark, 
                                         mock_get_fps, mock_unlink, mock_path_exists, 
                                         mock_copyfile, mock_os_system, mock_ffmpeg_input_module):
        
        mock_ffmpeg_run = MagicMock()
        mock_input_instance = MagicMock()
        mock_output_instance = MagicMock()
        mock_global_args_instance = MagicMock()

        mock_ffmpeg_input_module.return_value = mock_input_instance
        mock_input_instance.output.return_value = mock_output_instance
        mock_output_instance.global_args.return_value = mock_global_args_instance
        mock_global_args_instance.run = mock_ffmpeg_run

        source_path = Path('test_video.mp4')
        mock_get_fps.return_value = '30'
        
        audio_file_path = self.vp.audio_root / (source_path.stem + '.aac')
        colorized_no_audio_path = self.vp.result_folder / (source_path.stem + '_no_audio.mp4')
        expected_result_path = self.vp.result_folder / source_path.name
        
        # Path.exists side effect:
        os_system_call_count_for_exists_mock = 0
        def exists_side_effect_build(path_obj_being_checked):
            nonlocal os_system_call_count_for_exists_mock
            if path_obj_being_checked == colorized_no_audio_path: return False
            if path_obj_being_checked == expected_result_path: return False 
            if path_obj_being_checked == audio_file_path:
                # This is checked before and after extraction.
                # If os.system has been called at least once (for audio extraction), then it "exists".
                return os_system_call_count_for_exists_mock >= 1 
            return False 

        def os_system_side_effect_wrapper(command):
            nonlocal os_system_call_count_for_exists_mock
            os_system_call_count_for_exists_mock +=1
            return 0 # Success
            
        mock_path_exists.side_effect = exists_side_effect_build
        mock_os_system.side_effect = os_system_side_effect_wrapper
       
        # Call _build_video with watermarked=False
        result = self.vp._build_video(source_path, watermarked=False) 

        # Check mkdir calls
        mkdir_calls_made = {str(c[0][0]): c.kwargs for c_args, c_kwargs in mock_general_mkdir.call_args_list if c_args}
        
        self.assertIn(str(self.vp.audio_root), mkdir_calls_made)
        self.assertEqual(mkdir_calls_made[str(self.vp.audio_root)], {'parents': True, 'exist_ok': True})
        
        self.assertIn(str(colorized_no_audio_path.parent), mkdir_calls_made) # This is self.vp.result_folder
        self.assertEqual(mkdir_calls_made[str(colorized_no_audio_path.parent)], {'parents': True, 'exist_ok': True})


        colorframes_folder = self.vp.colorframes_root / source_path.stem
        colorframes_template = str(colorframes_folder / '%5d.jpg')
        mock_ffmpeg_input_module.assert_any_call(str(colorframes_template), format='image2', vcodec='mjpeg', framerate='30')

        mock_copyfile.assert_called_once_with(str(colorized_no_audio_path), str(expected_result_path))
       
        self.assertEqual(mock_os_system.call_count, 2) 
        mock_apply_watermark.assert_not_called() # Because watermarked=False
        self.assertEqual(result, expected_result_path)

if __name__ == '__main__':
    unittest.main()
