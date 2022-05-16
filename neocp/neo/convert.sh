for i in $(seq 0 199)
do
   printf -v two "%02d" $i
   printf -v three "%03d" $i
   mv night_${two}.filtered.dat night_${three}.filtered.dat
done
