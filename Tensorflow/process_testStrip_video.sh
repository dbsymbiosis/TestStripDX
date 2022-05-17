#!/usr/bin/env bash

#### Pre-run setup
#source ~/scripts/script_setup.sh
#set +eu; conda activate py27; set -eu

#### Get options
BLANK=0

usage() {
    echo -e "Usage: $(basename "$0") [OPTIONS...]
Options:
-v, --video      video file from test strip analysis.
-b, --blank      video is from blank (defalt=False; i.e., test video)
--debug          run debug mode" 1>&2
    exit 1
}

# See https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -v|--video)
      VIDEO="$2"
      shift # past argument
      shift # past value
      ;;
    -b|--blank)
      BLANK=1
      shift # past argument
      ;;
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


set +eu
if [ -z "${VIDEO}" ]; then
    usage
fi
set -eu

#### Setup env
#export PATH=$PATH:"${MMSEQS_BIN}"

## START: Check options set
if [ ! -s $VIDEO ]; then echo >&2 "[ERROR]: $VIDEO does not exist!"; exit 1; fi
## END: Check parsed options

## START: Check execuatble
FAIL_COUNT=0
for PROG in "python";
do 
	hash $PROG 2>/dev/null || { echo >&2 "[ERROR]: $PROG missing from PATH."; FAIL_COUNT=$((FAIL_COUNT+1)); }
done
if [ $FAIL_COUNT -gt 0 ]; then echo >&2 "...Aborting!"; exit 1; fi
## END: Check execuatble

# Setup variables
OUT="$VIDEO.TestStripDX"
mkdir -p "$OUT"


function extract_score () {
	VIDEO="${1}"
	SEC="${2}"
	OUT="${3}"
	CROP="${4}"
	NAME="${5}"
	FRAME="${OUT}/frame_${SEC}seconds"
	echo -e "Searching ${CROP} (${NAME}) - frame at time ${SEC} seconds"
	./extract_frame_from_timestamp.py --input_video "$VIDEO" --out_frame "${FRAME}.png" --seconds "$SEC"
	python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images "${FRAME}.png" --output "${FRAME}.detect" --crop --info --dont_show 1>"${FRAME}.detect.info.txt"
	SCORE=$(python ./extract_colors.py -i "${FRAME}.detect.crop/${CROP}_1.png")	
	echo -e "${NAME}\t$SCORE" > "${FRAME}.detect.crop/${CROP}_1.score.txt"
}


echo -e "########################################################"
echo -e "## Extracting frames from ${VIDEO}"
echo -e "########################################################"

if [ $BLANK -eq 0 ];
then
	echo -e "Running as test!"
	# Glucose: 	30sec
	# Ketone: 	40sec
	# Blood: 	60sec
	# Leukocytes:	120sec
	
	extract_score "$VIDEO" 30  "${OUT}" "Urobilinogen" "Glucose"
	extract_score "$VIDEO" 40  "${OUT}" "Protein"      "Ketone"
	extract_score "$VIDEO" 60  "${OUT}" "Nitrite"      "Blood"
	extract_score "$VIDEO" 120 "${OUT}" "pH"           "Leukocytes"

	cat ${OUT}/*/*.score.txt > "${OUT}/results.txt"
else
	echo -e "Running as blank!"
	# Glucose:      5sec
        # Ketone:       5sec
        # Blood:        5sec
        # Leukocytes:   5sec

        extract_score "$VIDEO" 5 "${OUT}" "Urobilinogen" "Glucose"
        extract_score "$VIDEO" 5 "${OUT}" "Protein"      "Ketone"
        extract_score "$VIDEO" 5 "${OUT}" "Nitrite"      "Blood"
        extract_score "$VIDEO" 5 "${OUT}" "pH"           "Leukocytes"
	
	cat ${OUT}/*/*.score.txt > "${OUT}/results.txt"
fi

echo -e "########################################################"
echo -e "## Finished. Results in ${OUT}/results.txt"
echo -e "########################################################"


