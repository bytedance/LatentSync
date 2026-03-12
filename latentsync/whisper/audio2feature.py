# Adapted from https://github.com/TMElyralab/MuseTalk/blob/main/musetalk/whisper/audio2feature.py

from .whisper import load_model
import numpy as np
import torch
import os
from pathlib import Path


class Audio2Feature:
    def __init__(
        self,
        model_path="checkpoints/whisper/tiny.pt",
        device=None,
        audio_embeds_cache_dir=None,
        num_frames=16,
        audio_feat_length=[2, 2],
    ):
        self.model = load_model(model_path, device)
        self.audio_embeds_cache_dir = audio_embeds_cache_dir
        if audio_embeds_cache_dir is not None and audio_embeds_cache_dir != "":
            Path(audio_embeds_cache_dir).mkdir(parents=True, exist_ok=True)
        self.num_frames = num_frames
        self.embedding_dim = self.model.dims.n_audio_state
        self.audio_feat_length = audio_feat_length

    def get_sliced_feature(self, feature_array, vid_idx, fps=25):
        """
        Get sliced features based on a given index
        :param feature_array:
        :param start_idx: the start index of the feature
        :param audio_feat_length:
        :return:
        """
        length = len(feature_array)
        selected_feature = []
        selected_idx = []

        center_idx = int(vid_idx * 50 / fps)
        left_idx = center_idx - self.audio_feat_length[0] * 2
        right_idx = center_idx + (self.audio_feat_length[1] + 1) * 2

        for idx in range(left_idx, right_idx):
            idx = max(0, idx)
            idx = min(length - 1, idx)
            x = feature_array[idx]
            selected_feature.append(x)
            selected_idx.append(idx)

        selected_feature = torch.cat(selected_feature, dim=0)
        selected_feature = selected_feature.reshape(-1, self.embedding_dim)  # 50*384
        return selected_feature, selected_idx

    def get_sliced_feature_sparse(self, feature_array, vid_idx, fps=25):
        """
        Get sliced features based on a given index
        :param feature_array:
        :param start_idx: the start index of the feature
        :param audio_feat_length:
        :return:
        """
        length = len(feature_array)
        selected_feature = []
        selected_idx = []

        for dt in range(-self.audio_feat_length[0], self.audio_feat_length[1] + 1):
            left_idx = int((vid_idx + dt) * 50 / fps)
            if left_idx < 1 or left_idx > length - 1:
                left_idx = max(0, left_idx)
                left_idx = min(length - 1, left_idx)

                x = feature_array[left_idx]
                x = x[np.newaxis, :, :]
                x = np.repeat(x, 2, axis=0)
                selected_feature.append(x)
                selected_idx.append(left_idx)
                selected_idx.append(left_idx)
            else:
                x = feature_array[left_idx - 1 : left_idx + 1]
                selected_feature.append(x)
                selected_idx.append(left_idx - 1)
                selected_idx.append(left_idx)
        selected_feature = np.concatenate(selected_feature, axis=0)
        selected_feature = selected_feature.reshape(-1, self.embedding_dim)  # 50*384
        selected_feature = torch.from_numpy(selected_feature)
        return selected_feature, selected_idx

    def feature2chunks(self, feature_array, fps):
        whisper_chunks = []
        whisper_idx_multiplier = 50.0 / fps
        i = 0
        print(f"video in {fps} FPS, audio idx in 50FPS")

        while True:
            start_idx = int(i * whisper_idx_multiplier)
            # Check boundary BEFORE appending to avoid generating extra frames
            if start_idx >= len(feature_array):
                break
            selected_feature, selected_idx = self.get_sliced_feature(feature_array=feature_array, vid_idx=i, fps=fps)
            # print(f"i:{i},selected_idx {selected_idx}")
            whisper_chunks.append(selected_feature)
            i += 1

        return whisper_chunks

    def _audio2feat(self, audio_path: str):
        # get the sample rate of the audio
        result = self.model.transcribe(audio_path)
        embed_list = []
        for emb in result["segments"]:
            encoder_embeddings = emb["encoder_embeddings"]
            encoder_embeddings = encoder_embeddings.transpose(0, 2, 1, 3)
            encoder_embeddings = encoder_embeddings.squeeze(0)
            start_idx = int(emb["start"])
            end_idx = int(emb["end"])
            emb_end_idx = int((end_idx - start_idx) / 2)
            embed_list.append(encoder_embeddings[:emb_end_idx])
        concatenated_array = torch.from_numpy(np.concatenate(embed_list, axis=0))
        return concatenated_array

    def audio2feat(self, audio_path):
        if self.audio_embeds_cache_dir == "" or self.audio_embeds_cache_dir is None:
            return self._audio2feat(audio_path)

        audio_embeds_cache_path = os.path.join(
            self.audio_embeds_cache_dir, os.path.basename(audio_path).replace(".mp4", "_embeds.pt")
        )

        if os.path.isfile(audio_embeds_cache_path):
            try:
                audio_feat = torch.load(audio_embeds_cache_path, weights_only=True)
            except Exception as e:
                print(f"{type(e).__name__} - {e} - {audio_embeds_cache_path}")
                os.remove(audio_embeds_cache_path)
                audio_feat = self._audio2feat(audio_path)
                torch.save(audio_feat, audio_embeds_cache_path)
        else:
            audio_feat = self._audio2feat(audio_path)
            torch.save(audio_feat, audio_embeds_cache_path)

        return audio_feat

    def crop_overlap_audio_window(self, audio_feat, start_index):
        selected_feature_list = []
        for i in range(start_index, start_index + self.num_frames):
            selected_feature, selected_idx = self.get_sliced_feature(feature_array=audio_feat, vid_idx=i, fps=25)
            selected_feature_list.append(selected_feature)
        mel_overlap = torch.stack(selected_feature_list)
        return mel_overlap


if __name__ == "__main__":
    audio_encoder = Audio2Feature(model_path="checkpoints/whisper/tiny.pt")
    audio_path = "assets/demo1_audio.wav"
    array = audio_encoder.audio2feat(audio_path)
    print(array.shape)
    fps = 25
    whisper_idx_multiplier = 50.0 / fps

    i = 0
    print(f"video in {fps} FPS, audio idx in 50FPS")
    while True:
        start_idx = int(i * whisper_idx_multiplier)
        # Check boundary BEFORE processing to avoid generating extra frames
        if start_idx >= len(array):
            break
        selected_feature, selected_idx = audio_encoder.get_sliced_feature(feature_array=array, vid_idx=i, fps=fps)
        print(f"video idx {i},\t audio idx {selected_idx},\t shape {selected_feature.shape}")
        i += 1
