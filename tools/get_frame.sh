/home/jovyan/ffmpeg/ffmpeg \
-i /home/jovyan/2024-srtp/srtp-final/giting-with-voice.mp4 \
-vf "select='between(n,21700,23700)'" \
 -vframes 400  ./ginting-2000.mp4