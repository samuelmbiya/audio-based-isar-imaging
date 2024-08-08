#!/bin/sh

while getopts "r:f:b:n:" flag;
do
    case ${flag} in
        r) range=${OPTARG};;
        f) center_freq=${OPTARG};;
        b) bandwidth=${OPTARG};;
        n) num_pulses=${OPTARG};;
    esac
done
echo "Range: $range m";
echo "Center_freq: $center_freq Hz";
echo "Bandwidth: $bandwidth Hz";
echo "Num Pulses: $num_pulses pulses";

# --range 3.0 --center_freq 7000.0 --bandwidth 860.0 --num_pulses 30
python3.11 record.py $range $center_freq & 
python3.11 chirp_burst.py "$range" "$center_freq" "$bandwidth" "$num_pulses" &
wait
python3.11 audio_processor.py "$range" "$center_freq" "$bandwidth"
# python3.11 chirp_burst.py $range $center_freq $bandwidth $num_pulses
# Requires PRI, Bandwidth, Center_Freq, Num_pulses