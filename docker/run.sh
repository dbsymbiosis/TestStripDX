

TAG="dev-StaticCoords_v0.0.2"


USER="timothystephens"
PROG="teststripdx"

# Build docker file from instructions
docker build -t $PROG:$TAG . 2>&1 | tee build.log

# Build docker image and push ro repo
BUILD=""

docker tag $BUILD $USER/$PROG:${TAG}
docker tag $BUILD $USER/$PROG:${TAG%.*}
docker tag $BUILD $USER/$PROG:${TAG%%.*}
#docker tag $BUILD $USER/teststripdx:latest

docker push --all-tags $USER/$PROG


# Test pull using singularity
conda activate singularity
IMG="$PROG:${TAG%%.*}"
singularity pull docker://$USER/$IMG

singularity run ${IMG}.sif process  -i ../test/*.mp4
singularity run ${IMG}.sif combine  -i ../test/*.results.txt   -o ../test/combined.results.txt
singularity run ${IMG}.sif joinPDFs -i ../test/*.detection.pdf -o ../test/combined.detection.pdf

# Test with docker - probabily wont work since docker is running as 
#	root, and this means that we cant create file in the users 
#	directiry as we are the wrong user
docker run --workdir=$PWD ${IMG} process  -i test/*.mp4
docker run --workdir=$PWD ${IMG} combine  -i ../test/*.results.txt   -o ../test/combined.results.txt
docker run --workdir=$PWD ${IMG} joinPDFs -i ../test/*.detection.pdf -o ../test/combined.detection.pdf


