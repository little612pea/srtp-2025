# This script is used to get the action of the video
id=$1
cd /home/jovyan/2024-srtp/srtp-final/mmaction2/
python get_pkl_from_hitframe.py --json_path /home/jovyan/2024-srtp/srtp-final/hit_frame_detection/outputs/joints/input_video/rally_${id}.json \
  --output_path output_hitting.pkl
python test.py /home/jovyan/2024-srtp/srtp-final/mmaction2/configs/skeleton/posec3d/custom_slowonly_ntu60.py ckpt/epoch_2.pth --dump temp.pkl
python get_txt_json.py
python get_final_action.py