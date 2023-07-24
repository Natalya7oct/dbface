from __future__ import annotations

import pytest

from dbface_package import main_openvino


@pytest.mark.load_video('videos/1.mp4')
def test1(video):
    main_openvino.camera_demo(video)


@pytest.mark.load_image('videos/2.mp4')
def test2(video):
    main_openvino.camera_demo(video)


@pytest.mark.load_image('videos/3.mp4')
def test3(video):
    main_openvino.camera_demo(video)
