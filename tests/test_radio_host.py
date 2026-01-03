import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
from pydub import AudioSegment

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import radio_host_functions as rhf

CONFIG = rhf.CONFIG


class TestCleanForTTS:
    def test_removes_speaker_labels_and_parentheses(self):
        assert rhf.clean_for_tts("Vijay: Hello (laughs) world") == "Hello world"
        assert rhf.clean_for_tts("Neha: Test message") == "Test message"
        assert rhf.clean_for_tts("Regular text") == "Regular text"


class TestGenerateScriptPrompt:
    def test_prompt_contains_required_elements(self):
        prompt = rhf.generate_script_prompt("Test topic")
        assert "Test topic" in prompt
        assert "Vijay" in prompt
        assert "Neha" in prompt
        assert "Hinglish" in prompt


class TestFetchWikipediaArticle:
    @patch('radio_host_functions.wikipediaapi.Wikipedia')
    def test_successful_fetch(self, mock_wikipedia):
        mock_page = Mock()
        mock_page.exists.return_value = True
        mock_page.text = "A" * 500
        mock_wiki = Mock()
        mock_wiki.page.return_value = mock_page
        mock_wikipedia.return_value = mock_wiki
        
        result = rhf.fetch_wikipedia_article("Test")
        assert len(result) == 500
    
    @patch('radio_host_functions.wikipediaapi.Wikipedia')
    def test_errors(self, mock_wikipedia):
        mock_page = Mock()
        mock_page.exists.return_value = False
        mock_wiki = Mock()
        mock_wiki.page.return_value = mock_page
        mock_wikipedia.return_value = mock_wiki
        
        with pytest.raises(ValueError, match="Wikipedia page not found"):
            rhf.fetch_wikipedia_article("Nonexistent")


class TestGenerateScript:
    def test_successful_generation(self):
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Vijay: Hello\nNeha: Hi\n"
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        result = rhf.generate_script("prompt", mock_client)
        assert "Vijay" in result
        assert mock_client.chat.completions.create.called
    
    def test_empty_script_error(self):
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = ""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        with pytest.raises(RuntimeError, match="Empty script"):
            rhf.generate_script("prompt", mock_client)


class TestGenerateAudioSegments:
    @patch.dict('os.environ', {'ELEVENLABS_API_KEY': 'test_key'})
    @patch('radio_host_functions.ElevenLabs')
    @patch('radio_host_functions.AudioSegment.silent')
    @patch('radio_host_functions.AudioSegment.from_file')
    def test_generates_audio(self, mock_from_file, mock_silent, mock_elevenlabs):
        mock_segment = Mock()
        mock_silent.return_value = Mock()
        mock_segment.__add__ = Mock(return_value=mock_segment)
        mock_from_file.return_value = mock_segment
        
        mock_client = Mock()
        mock_client.text_to_speech.convert.return_value = [b"audio"]
        mock_elevenlabs.return_value = mock_client
        
        result = rhf.generate_audio_segments("Vijay: Hello\nNeha: Hi\n")
        assert len(result) == 2
    
    @patch.dict('os.environ', {}, clear=True)
    def test_missing_api_key(self):
        with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY not set"):
            rhf.generate_audio_segments("Vijay: Test\n")


class TestCombineAndExportAudio:
    @patch('radio_host_functions.os.path.getsize')
    def test_combines_and_exports(self, mock_getsize):
        # Create mock audio segments
        mock_seg1 = Mock(spec=AudioSegment)
        mock_seg1.__len__ = Mock(return_value=1000)
        mock_seg2 = Mock(spec=AudioSegment)
        mock_seg2.__len__ = Mock(return_value=2000)
        
        mock_final = Mock(spec=AudioSegment)
        mock_final.__len__ = Mock(return_value=3000)
        mock_final.normalize = Mock(return_value=mock_final)
        mock_final.export = Mock()
        mock_seg1.__add__ = Mock(return_value=mock_final)
        
        mock_getsize.return_value = 1024 * 1024
        
        segments = [mock_seg1, mock_seg2]
        rhf.combine_and_export_audio(segments, "test.mp3")
        
        mock_final.normalize.assert_called_once_with(headroom=1.5)
        mock_final.export.assert_called_once_with("test.mp3", format="mp3", bitrate="192k")
