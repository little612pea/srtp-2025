/home/jovyan/ffmpeg/ffmpeg \
-i /home/jovyan/2024-srtp/CollectInfo/SPOT/data/tennis/videos/Anders_Antonsen_Viktor_Axelsen_HSBC_BWF_WORLD_TOUR_FINALS_2020_Finals/Anders_Antonsen_Viktor_Axelsen_HSBC_BWF_WORLD_TOUR_FINALS_2020_Finals.mp4 \
-vf "select='between(n,21700,23700)'" \
 -vframes 400  ./alex-2000.mp4