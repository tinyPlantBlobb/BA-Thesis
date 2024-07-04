
#example use from danni on the segmenter
sl=”en”
tl=”de”
src_xml=”IWSLT.TED.tst2020.en-de.en.xml”
ref_xml=”IWSLT.TED.tst2020.en-de.de.xml”
ref=”en-de.ref”
name=”system1”

# run mwer segmentation
~/opt/mwerSegmenter/segmentBasedOnMWER.sh ${src_xml} ${ref_xml} $EVAL_DIR/$sl-$tl.sys $name ${tl} $EVAL_DIR/$sl-$tl.sgm no_normalize 1

# post-process produced sgm file
sed -e "/<[^>]*>/d" $EVAL_DIR/$sl-$tl.sgm > $EVAL_DIR/$sl-$tl.aligned.sys

# evaluate against reference with sacreBLEU
cat $EVAL_DIR/$sl-$tl.aligned.sys | sacrebleu $ref -m chrf bleu -tok zh > $EVAL_DIR/$sl-$tl.res
