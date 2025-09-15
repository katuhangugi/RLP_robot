ffmpeg -i /home/joe/Work/RPL_local/001_1.mkv -filter_complex \
"[0:v]scale=ih*4/3:ih,setsar=1[v1]; \
 [0:v]scale=ih*16/9:ih,boxblur=10:1[bg]; \
 [bg][v1]overlay=(W-w)/2:0" \
-c:a copy output_final.mkv
