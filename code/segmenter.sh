#example use from danni on the segmenter
sl=en
tl=de
GENBASE=~/uni/BA/BA-Thesis/code
REFBASE=$GENBASE/IWSLT23.tst2023.en-de/benchmark/en-de/tst2023
src_xml=$REFBASE/IWSLT.TED.tst2023.en-de.en.xml
ref_xml=$REFBASE/IWSLT.TED.tst2023.en-de.de.xml
ref=en-de.ref
name=”whisper-seamless”
if [ -z "$1" ]; then
  MODELTRANSLATION=data-bin/test.de
else
  MODELTRANSLATION=$1
fi

EVAL_DIR=$GENBASE/evaluation

# run mwer segmentation
$GENBASE/mwerSegmenter/segmentBasedOnMWER.sh ${src_xml} ${ref_xml} $MODELTRANSLATION $name ${tl} $EVAL_DIR/$sl-$tl.sgm no_normalize 1

# post-process produced sgm file
sed -e "/<[^>]*>/d" $EVAL_DIR/$sl-$tl.sgm >$EVAL_DIR/$sl-$tl.aligned.sys

# evaluate against reference with sacreBLEU
#cat $EVAL_DIR/$sl-$tl.aligned.sys | sacrebleu $ref -m chrf bleu -tok zh >$EVAL_DIR/$sl-$tl.res
