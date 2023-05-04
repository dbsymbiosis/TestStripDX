

TAG="dev-StaticCoords_v0.0.7"

USER="timothystephens"
PROG="teststripdx"



## Build docker file from instructions
#  use '--no-cache' to ensure the github repo is always re-downloaded when building
cp -r ../models .
docker build --no-cache -t $PROG:$TAG . 2>&1 | tee build.log
rm -r models



## Test that image runs as expected.
docker run --workdir=$PWD $PROG:${TAG} -h



## Build docker image and push ro repo
BUILD=$(awk '$1=="Successfully" && $2=="built" {print $3}' build.log)
echo "BUILD=${BUILD}"

docker tag $BUILD $USER/$PROG:${TAG}
docker tag $BUILD $USER/$PROG:${TAG%.*}
docker tag $BUILD $USER/$PROG:${TAG%%.*}
#docker tag $BUILD $USER/teststripdx:latest

docker push --all-tags $USER/$PROG



## Test pull using singularity
conda activate singularity
IMG="${PROG}_${TAG%%.*}"
singularity pull docker://$USER/${PROG}:${TAG%%.*}
singularity run ${IMG}.sif process  -i ../test/vid*.mp4 1>test_process.log 2>&1
singularity run ${IMG}.sif combine  -i ../test/vid*.results.txt   -o ../test/combined.results.txt 1>test_combine.log 2>&1
singularity run ${IMG}.sif joinPDFs -i ../test/vid*.detection.pdf -o ../test/combined.detection.pdf 1>test_joinPDFs.log 2>&1



## Test with docker
#    - probabily wont work since docker is running as 
#      root, and this means that we cant create file in the users 
#      directiry as we are the wrong user
IMG="${USER}/${PROG}:${TAG%%.*}"
(
cd ../test
docker run --workdir=$PWD ${IMG} process  -i vid*.mp4
docker run --workdir=$PWD ${IMG} combine  -i vid*.results.txt   -o combined.results.txt
docker run --workdir=$PWD ${IMG} joinPDFs -i vid*.detection.pdf -o combined.detection.pdf
)


## Cleanup docker, singularity, and test data
docker image prune -af
docker image rm -f $(docker image ls | awk 'NR>1{print $3}')
rm -fr *.sif
rm -fr ../test/*TestStripDX* ../test/combined.*



