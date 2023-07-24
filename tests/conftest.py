from __future__ import annotations

import os

import cv2
import pytest


@pytest.fixture(name='video')
def video(request):
    test_path = request.node.path
    test_dir_path = os.path.dirname(test_path)

    marker = request.node.get_closest_marker('load_video')
    relative_video_path = marker.args[0]

    video_path = os.path.join(test_dir_path, relative_video_path)

    yield cv2.imread(video_path)
