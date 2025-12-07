import sys
import unittest
from pathlib import Path

import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from indextts.infer_v2 import IndexTTS2


class TestStreamingEndToEnd(unittest.TestCase):
    def test_streaming_inference_real(self):
        # Ensure we are in the root directory or paths are correct
        root_dir = Path(__file__).parent.parent
        checkpoint_dir = root_dir / "checkpoints"
        config_path = checkpoint_dir / "config.yaml"

        if not checkpoint_dir.exists():
            self.skipTest("Checkpoints directory not found")

        # Create a dummy wav file for testing
        prompt_wav = root_dir / "tests" / "temp_test_prompt.wav"
        sample_rate = 16000
        # Generate 1 second of silence/noise
        dummy_audio = torch.randn(1, sample_rate)
        import torchaudio

        torchaudio.save(prompt_wav, dummy_audio, sample_rate)

        try:
            # Initialize real model
            # Use CPU for testing to ensure it runs everywhere, or check availability
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")

            # Initialize the TTS engine
            # We disable some optimizations to make initialization faster/safer for testing
            tts = IndexTTS2(
                cfg_path=config_path, model_dir=checkpoint_dir, device=device, use_fp16=False, use_torch_compile=False
            )

            text = "Hello world. This is a test of streaming inference."

            print(f"Starting inference with text: '{text}'")

            # Run inference with streaming
            # We set max_text_tokens_per_segment to a small number to encourage segmentation
            generator = tts.infer(
                spk_audio_prompt=prompt_wav,
                text=text,
                output_path=None,
                stream_return=True,
                max_text_tokens_per_segment=10,
                verbose=True,
            )

            # Verify it returns a generator
            self.assertTrue(hasattr(generator, "__iter__"))

            chunks = []
            print("Consuming generator...")
            for i, chunk in enumerate(generator):
                print(f"Received chunk {i}, shape: {chunk.shape}")
                self.assertIsInstance(chunk, torch.Tensor)
                # Check that the chunk is not empty
                self.assertTrue(chunk.numel() > 0)
                chunks.append(chunk)

            self.assertTrue(len(chunks) > 0)
            print(f"Total chunks received: {len(chunks)}")

            # Concatenate chunks to verify full audio structure
            full_audio = torch.cat(chunks, dim=-1)
            print(f"Full audio shape: {full_audio.shape}")
            self.assertTrue(full_audio.shape[-1] > 0)

        finally:
            # Cleanup
            if prompt_wav.exists():
                prompt_wav.unlink()


if __name__ == "__main__":
    unittest.main()
