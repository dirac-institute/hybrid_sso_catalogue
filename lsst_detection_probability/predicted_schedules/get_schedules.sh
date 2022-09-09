#!/bin/bash

FILES_TO_COPY=""
for i in {1..365}
do
    FILES_TO_COPY+="/gscratch/scrubbed/yoachim/survey_predicted/night${i}_15days.db "
done

echo $FILES_TO_COPY

scp tomwagg@mox.hyak.uw.edu:"$FILES_TO_COPY" .
