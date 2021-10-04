echo "downloading dataset..."

kaggle datasets download mozillaorg/common-voice

echo "done downloading dataset.\n"
echo "extracting dataset..."

unzip -q common-voice.zip

echo "done extracting dataset.\n"
echo "cleaning up..."

rm common-voice.zip LICENSE.txt README.txt

mkdir ./cv-valid-train/clips/
mv ./cv-valid-train/cv-valid-train ./cv-valid-train/clips/

mkdir ./cv-valid-test/clips/
mv ./cv-valid-test/cv-valid-test ./cv-valid-test/clips/

rm -r cv-other-dev cv-other-test cv-other-train cv-valid-dev cv-invalid
rm cv-valid-dev.csv cv-other-dev.csv cv-other-test.csv cv-other-train.csv cv-invalid.csv

echo "done preparing dataset.\n"