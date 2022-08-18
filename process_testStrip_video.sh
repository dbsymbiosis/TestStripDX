#!/usr/bin/env bash

#### Get options
usage() {
    echo -e "Usage: $(basename "$0") [OPTIONS...] VIDEO [VIDEO ..]
Options:
--debug          run debug mode" 1>&2
    exit 1
}

# See https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --debug)
      set -x
      shift # past argument
      ;;
    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters


# Print usage if no videos provided
set +eu
if [ "${#POSITIONAL[@]}" -eq 0 ]; then
    usage
fi
set -eu

#SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
#export PATH="$PATH:$SCRIPT_DIR"

## START: Check execuatble
FAIL_COUNT=0
for PROG in "python";
do 
	hash $PROG 2>/dev/null || { echo >&2 "[ERROR]: $PROG missing from PATH."; FAIL_COUNT=$((FAIL_COUNT+1)); }
done
if [ $FAIL_COUNT -gt 0 ]; then echo >&2 "...Aborting!"; exit 1; fi
## END: Check execuatble


## Functions
function extract_score () {
	VIDEO="${1}"
	SEC="${2}"
	OUT="${3}"
	CROP="${4}"
	NAME="${5}"
	FRAME="${OUT}/frame.${SEC}sec"
	echo -e "Searching ${CROP} (${NAME}) - frame at time ${SEC} seconds"
	python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images "${FRAME}.png" --output "${FRAME}.detect" --crop --info --dont_show 1>"${FRAME}.detect.info.txt"
	if [ -s "${FRAME}.detect.crop/${CROP}_1.png" ];
	then
		SCORE=$(python ./extract_colors.py -i "${FRAME}.detect.crop/${CROP}_1.png")	
		echo -e "${NAME}\t${SCORE}" > "${FRAME}.detect.crop/${CROP}_1.score.txt"
	else
		echo "[WARNING] No frame detected for ${CROP} (${NAME}) - frame at time ${SEC} seconds"
		echo -e "${NAME}\tNA" > "${FRAME}.detect.crop/${CROP}_1.score.txt"
	fi
}


## Loop over videos
for VIDEO in "${POSITIONAL[@]}";
do
	echo -e "########################################################"
	echo -e "## Extracting frames from ${VIDEO}"
	echo -e "########################################################"
	
	OUT="$VIDEO.TestStripDX"
	rm -rf "$OUT"
	mkdir -p "$OUT"

	# Glucose: 	30sec
	# Ketone: 	40sec
	# Blood: 	60sec
	# Leukocytes:	120sec
	
	python extract_frame_from_timestamp.py --input_video "$VIDEO" --out_prefix "${OUT}/frame" --seconds 30 40 60 118
	extract_score "$VIDEO" 30  "${OUT}" "Urobilinogen" "Glucose"
	extract_score "$VIDEO" 40  "${OUT}" "Protein"      "Ketone"
	extract_score "$VIDEO" 60  "${OUT}" "Nitrite"      "Blood"
	extract_score "$VIDEO" 118 "${OUT}" "pH"           "Leukocytes"
	
	cat ${OUT}/*/*.score.txt > "${OUT}/results.txt"
	
	echo -e "########################################################"
	echo -e "## Finished. Results in ${OUT}/results.txt"
	echo -e "########################################################"
done



