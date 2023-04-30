./TestStripDX.py process -m URS10 -v /data/*.mp4
./TestStripDX.py combine -m URS10 -t /data/*.TestStripDX.results.txt -o /data/combined_results.txt
./TestStripDX.py joinPDFs -o /data/merged.pdf -i /data/*.TestStripDX.detection.pdf
