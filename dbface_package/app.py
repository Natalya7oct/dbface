import dbface_lib
import os
import streamlit as st

def write_bytesio_to_file(filename, bytesio):
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())

#video_path = os.path.normpath(os.getcwd() + '/test_video.mp4')
#model_path = os.path.normpath(os.getcwd() + '/FP16/dbface.xml')
default_video = './test_video.mp4'
model_path = './FP16/dbface.xml'

st.title("Welcome to :blue[DBFace] demo!")
st.divider()
st.write('DBFace is a real-time, single-stage detector for face detection, with faster speed and higher accuracy.')
st.write('Now you see result for test video, but you can try if for your video.')
st.write(':sunglasses: Try it, choose mp4 file:')
input_video = st.file_uploader("", accept_multiple_files=False)
temp_file = './temp_video.mp4'


try:
    write_bytesio_to_file(temp_file, input_video)
    dbface_lib.camera_demo(video=temp_file, model_xml=model_path)
except:
    dbface_lib.camera_demo(video=default_video, model_xml=model_path)

