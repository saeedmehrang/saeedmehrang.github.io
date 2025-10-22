Segment Anything

Alexander Kirillov1,2,4
Tete Xiao3

Eric Mintun2 Nikhila Ravi1,2 Hanzi Mao2 Chloe Rolland3
Piotr Doll´ar4
4directional lead

Alexander C. Berg Wan-Yen Lo

3equal contribution

2joint ﬁrst author

Spencer Whitehead
1project lead

Laura Gustafson3
Ross Girshick4

Meta AI Research, FAIR

Figure 1: We aim to build a foundation model for segmentation by introducing three interconnected components: a prompt-
able segmentation task, a segmentation model (SAM) that powers data annotation and enables zero-shot transfer to a range
of tasks via prompt engineering, and a data engine for collecting SA-1B, our dataset of over 1 billion masks.

Abstract

We introduce the Segment Anything (SA) project: a new
task, model, and dataset for image segmentation. Using our
efﬁcient model in a data collection loop, we built the largest
segmentation dataset to date (by far), with over 1 billion
masks on 11M licensed and privacy respecting images. The
model is designed and trained to be promptable, so it can
transfer zero-shot to new image distributions and tasks. We
evaluate its capabilities on numerous tasks and ﬁnd that
its zero-shot performance is impressive – often competitive
with or even superior to prior fully supervised results. We
are releasing the Segment Anything Model (SAM) and cor-
responding dataset (SA-1B) of 1B masks and 11M images at
https://segment-anything.com to foster research into foun-
dation models for computer vision.

1. Introduction

Large language models pre-trained on web-scale datasets
are revolutionizing NLP with strong zero-shot and few-shot
generalization [10]. These “foundation models” [8] can
generalize to tasks and data distributions beyond those seen
during training. This capability is often implemented with
prompt engineering in which hand-crafted text is used to
prompt the language model to generate a valid textual re-
sponse for the task at hand. When scaled and trained with
abundant text corpora from the web, these models’ zero and
few-shot performance compares surprisingly well to (even

matching in some cases) ﬁne-tuned models [10, 21]. Empir-
ical trends show this behavior improving with model scale,
dataset size, and total training compute [56, 10, 21, 51].

Foundation models have also been explored in computer
vision, albeit to a lesser extent. Perhaps the most promi-
nent illustration aligns paired text and images from the web.
For example, CLIP [82] and ALIGN [55] use contrastive
learning to train text and image encoders that align the two
modalities. Once trained, engineered text prompts enable
zero-shot generalization to novel visual concepts and data
distributions. Such encoders also compose effectively with
other modules to enable downstream tasks, such as image
generation (e.g., DALL·E [83]). While much progress has
been made on vision and language encoders, computer vi-
sion includes a wide range of problems beyond this scope,
and for many of these, abundant training data does not exist.
In this work, our goal is to build a foundation model for
image segmentation. That is, we seek to develop a prompt-
able model and pre-train it on a broad dataset using a task
that enables powerful generalization. With this model, we
aim to solve a range of downstream segmentation problems
on new data distributions using prompt engineering.

The success of this plan hinges on three components:
task, model, and data. To develop them, we address the
following questions about image segmentation:

1. What task will enable zero-shot generalization?
2. What is the corresponding model architecture?
3. What data can power this task and model?

1

(b) Model: Segment Anything Model (SAM)promptimagevalid maskimage encoderprompt encoderlightweight mask decoder(a) Task: promptable segmentationsegmentation promptimagemodelcat withblack earsvalid mask(c) Data: data engine (top) & dataset (bottom)•1+ billion masks•11 million images •privacy respecting•licensed imagesannotatetraindatamodelSegment Anything 1B (SA-1B):

These questions are entangled and require a comprehen-
sive solution. We start by deﬁning a promptable segmenta-
tion task that is general enough to provide a powerful pre-
training objective and to enable a wide range of downstream
applications. This task requires a model that supports ﬂex-
ible prompting and can output segmentation masks in real-
time when prompted to allow for interactive use. To train
our model, we need a diverse, large-scale source of data.
Unfortunately, there is no web-scale data source for seg-
mentation; to address this, we build a “data engine”, i.e.,
we iterate between using our efﬁcient model to assist in data
collection and using the newly collected data to improve the
model. We introduce each interconnected component next,
followed by the dataset we created and the experiments that
demonstrate the effectiveness of our approach.

Task (§2).
In NLP and more recently computer vision,
foundation models are a promising development that can
perform zero-shot and few-shot learning for new datasets
and tasks often by using “prompting” techniques. Inspired
by this line of work, we propose the promptable segmen-
tation task, where the goal is to return a valid segmenta-
tion mask given any segmentation prompt (see Fig. 1a). A
prompt simply speciﬁes what to segment in an image, e.g.,
a prompt can include spatial or text information identifying
an object. The requirement of a valid output mask means
that even when a prompt is ambiguous and could refer to
multiple objects (for example, a point on a shirt may in-
dicate either the shirt or the person wearing it), the output
should be a reasonable mask for at least one of those ob-
jects. We use the promptable segmentation task as both a
pre-training objective and to solve general downstream seg-
mentation tasks via prompt engineering.

Model (§3). The promptable segmentation task and the goal
of real-world use impose constraints on the model architec-
ture. In particular, the model must support ﬂexible prompts,
needs to compute masks in amortized real-time to allow in-
teractive use, and must be ambiguity-aware. Surprisingly,
we ﬁnd that a simple design satisﬁes all three constraints:
a powerful image encoder computes an image embedding,
a prompt encoder embeds prompts, and then the two infor-
mation sources are combined in a lightweight mask decoder
that predicts segmentation masks. We refer to this model as
the Segment Anything Model, or SAM (see Fig. 1b). By
separating SAM into an image encoder and a fast prompt
encoder / mask decoder, the same image embedding can
be reused (and its cost amortized) with different prompts.
Given an image embedding, the prompt encoder and mask
decoder predict a mask from a prompt in ∼50ms in a web
browser. We focus on point, box, and mask prompts, and
also present initial results with free-form text prompts. To
make SAM ambiguity-aware, we design it to predict mul-
tiple masks for a single prompt allowing SAM to naturally
handle ambiguity, such as the shirt vs. person example.

Data engine (§4). To achieve strong generalization to new
data distributions, we found it necessary to train SAM on
a large and diverse set of masks, beyond any segmenta-
tion dataset that already exists. While a typical approach
for foundation models is to obtain data online [82], masks
are not naturally abundant and thus we need an alternative
strategy. Our solution is to build a “data engine”, i.e., we
co-develop our model with model-in-the-loop dataset an-
notation (see Fig. 1c). Our data engine has three stages:
assisted-manual, semi-automatic, and fully automatic.
In
the ﬁrst stage, SAM assists annotators in annotating masks,
similar to a classic interactive segmentation setup. In the
second stage, SAM can automatically generate masks for
a subset of objects by prompting it with likely object lo-
cations and annotators focus on annotating the remaining
objects, helping increase mask diversity. In the ﬁnal stage,
we prompt SAM with a regular grid of foreground points,
yielding on average ∼100 high-quality masks per image.

Dataset (§5). Our ﬁnal dataset, SA-1B, includes more than
1B masks from 11M licensed and privacy-preserving im-
ages (see Fig. 2). SA-1B, collected fully automatically us-
ing the ﬁnal stage of our data engine, has 400× more masks
than any existing segmentation dataset [66, 44, 117, 60],
and as we verify extensively, the masks are of high quality
and diversity. Beyond its use in training SAM to be robust
and general, we hope SA-1B becomes a valuable resource
for research aiming to build new foundation models.

Responsible AI (§6). We study and report on potential fair-
ness concerns and biases when using SA-1B and SAM. Im-
ages in SA-1B span a geographically and economically di-
verse set of countries and we found that SAM performs sim-
ilarly across different groups of people. Together, we hope
this will make our work more equitable for real-world use
cases. We provide model and dataset cards in the appendix.

Experiments (§7). We extensively evaluate SAM. First, us-
ing a diverse new suite of 23 segmentation datasets, we ﬁnd
that SAM produces high-quality masks from a single fore-
ground point, often only slightly below that of the manu-
ally annotated ground truth. Second, we ﬁnd consistently
strong quantitative and qualitative results on a variety of
downstream tasks under a zero-shot transfer protocol using
prompt engineering, including edge detection, object pro-
posal generation, instance segmentation, and a preliminary
exploration of text-to-mask prediction. These results sug-
gest that SAM can be used out-of-the-box with prompt en-
gineering to solve a variety of tasks involving object and
image distributions beyond SAM’s training data. Neverthe-
less, room for improvement remains, as we discuss in §8.

Release. We are releasing the SA-1B dataset for research
purposes and making SAM available under a permissive
open license (Apache 2.0) at https://segment-anything.com.
We also showcase SAM’s capabilities with an online demo.

2

s
k
s
a
m
0
5

<

s
k
s
a
m
0
0
1
-
0
5

s
k
s
a
m
0
0
2
-
0
0
1

s
k
s
a
m
0
0
3
-
0
0
2

s
k
s
a
m
0
0
4
-
0
0
3

s
k
s
a
m
0
0
5
-
0
0
4

s
k
s
a
m
0
0
5

>

Figure 2: Example images with overlaid masks from our newly introduced dataset, SA-1B. SA-1B contains 11M diverse,
high-resolution, licensed, and privacy protecting images and 1.1B high-quality segmentation masks. These masks were
annotated fully automatically by SAM, and as we verify by human ratings and numerous experiments, are of high quality and
diversity. We group images by number of masks per image for visualization (there are ∼100 masks per image on average).

3

2. Segment Anything Task

We take inspiration from NLP, where the next token pre-
diction task is used for foundation model pre-training and
to solve diverse downstream tasks via prompt engineer-
ing [10]. To build a foundation model for segmentation,
we aim to deﬁne a task with analogous capabilities.

Task. We start by translating the idea of a prompt from NLP
to segmentation, where a prompt can be a set of foreground
/ background points, a rough box or mask, free-form text,
or, in general, any information indicating what to segment
in an image. The promptable segmentation task, then, is to
return a valid segmentation mask given any prompt. The re-
quirement of a “valid” mask simply means that even when
a prompt is ambiguous and could refer to multiple objects
(e.g., recall the shirt vs. person example, and see Fig. 3),
the output should be a reasonable mask for at least one of
those objects. This requirement is similar to expecting a lan-
guage model to output a coherent response to an ambiguous
prompt. We choose this task because it leads to a natural
pre-training algorithm and a general method for zero-shot
transfer to downstream segmentation tasks via prompting.

Pre-training. The promptable segmentation task suggests a
natural pre-training algorithm that simulates a sequence of
prompts (e.g., points, boxes, masks) for each training sam-
ple and compares the model’s mask predictions against the
ground truth. We adapt this method from interactive seg-
mentation [109, 70], although unlike interactive segmenta-
tion whose aim is to eventually predict a valid mask after
enough user input, our aim is to always predict a valid mask
for any prompt even when the prompt is ambiguous. This
ensures that a pre-trained model is effective in use cases that
involve ambiguity, including automatic annotation as re-
quired by our data engine §4. We note that performing well
at this task is challenging and requires specialized modeling
and training loss choices, which we discuss in §3.

Zero-shot transfer.
Intuitively, our pre-training task en-
dows the model with the ability to respond appropriately to
any prompt at inference time, and thus downstream tasks
can be solved by engineering appropriate prompts. For ex-
ample, if one has a bounding box detector for cats, cat in-
stance segmentation can be solved by providing the detec-
tor’s box output as a prompt to our model. In general, a wide
array of practical segmentation tasks can be cast as prompt-
ing. In addition to automatic dataset labeling, we explore
ﬁve diverse example tasks in our experiments in §7.

Related tasks. Segmentation is a broad ﬁeld: there’s in-
teractive segmentation [57, 109], edge detection [3], su-
per pixelization [85], object proposal generation [2], fore-
ground segmentation [94], semantic segmentation [90], in-
stance segmentation [66], panoptic segmentation [59], etc.
The goal of our promptable segmentation task is to produce

Figure 3: Each column shows 3 valid masks generated by
SAM from a single ambiguous point prompt (green circle).

a broadly capable model that can adapt to many (though
not all) existing and new segmentation tasks via prompt
engineering. This capability is a form of task generaliza-
tion [26]. Note that this is different than previous work on
multi-task segmentation systems. In a multi-task system, a
single model performs a ﬁxed set of tasks, e.g., joint seman-
tic, instance, and panoptic segmentation [114, 19, 54], but
the training and test tasks are the same. An important dis-
tinction in our work is that a model trained for promptable
segmentation can perform a new, different task at inference
time by acting as a component in a larger system, e.g., to
perform instance segmentation, a promptable segmentation
model is combined with an existing object detector.

Discussion. Prompting and composition are powerful tools
that enable a single model to be used in extensible ways, po-
tentially to accomplish tasks unknown at the time of model
design. This approach is analogous to how other founda-
tion models are used, e.g., how CLIP [82] is the text-image
alignment component of the DALL·E [83] image generation
system. We anticipate that composable system design, pow-
ered by techniques such as prompt engineering, will enable
a wider variety of applications than systems trained specif-
ically for a ﬁxed set of tasks. It’s also interesting to com-
pare promptable and interactive segmentation through the
lens of composition: while interactive segmentation mod-
els are designed with human users in mind, a model trained
for promptable segmentation can also be composed into a
larger algorithmic system as we will demonstrate.

4

Figure 4: Segment Anything Model (SAM) overview. A heavyweight image encoder outputs an image embedding that can
then be efﬁciently queried by a variety of input prompts to produce object masks at amortized real-time speed. For ambiguous
prompts corresponding to more than one object, SAM can output multiple valid masks and associated conﬁdence scores.

3. Segment Anything Model

We next describe the Segment Anything Model (SAM)
for promptable segmentation. SAM has three components,
illustrated in Fig. 4: an image encoder, a ﬂexible prompt
encoder, and a fast mask decoder. We build on Transformer
vision models [14, 33, 20, 62] with speciﬁc tradeoffs for
(amortized) real-time performance. We describe these com-
ponents at a high-level here, with details in §A.

Image encoder. Motivated by scalability and powerful pre-
training methods, we use an MAE [47] pre-trained Vision
Transformer (ViT) [33] minimally adapted to process high
resolution inputs [62]. The image encoder runs once per
image and can be applied prior to prompting the model.

Prompt encoder. We consider two sets of prompts: sparse
text) and dense (masks). We represent
(points, boxes,
points and boxes by positional encodings [95] summed with
learned embeddings for each prompt type and free-form text
with an off-the-shelf text encoder from CLIP [82]. Dense
prompts (i.e., masks) are embedded using convolutions and
summed element-wise with the image embedding.

Mask decoder. The mask decoder efﬁciently maps the im-
age embedding, prompt embeddings, and an output token
to a mask. This design, inspired by [14, 20], employs a
modiﬁcation of a Transformer decoder block [103] followed
by a dynamic mask prediction head. Our modiﬁed decoder
block uses prompt self-attention and cross-attention in two
directions (prompt-to-image embedding and vice-versa) to
update all embeddings. After running two blocks, we up-
sample the image embedding and an MLP maps the output
token to a dynamic linear classiﬁer, which then computes
the mask foreground probability at each image location.

Resolving ambiguity. With one output, the model will av-
erage multiple valid masks if given an ambiguous prompt.
To address this, we modify the model to predict multiple
output masks for a single prompt (see Fig. 3). We found
3 mask outputs is sufﬁcient to address most common cases
(nested masks are often at most three deep: whole, part, and
subpart). During training, we backprop only the minimum

loss [15, 45, 64] over masks. To rank masks, the model pre-
dicts a conﬁdence score (i.e., estimated IoU) for each mask.

Efﬁciency. The overall model design is largely motivated
by efﬁciency. Given a precomputed image embedding, the
prompt encoder and mask decoder run in a web browser, on
CPU, in ∼50ms. This runtime performance enables seam-
less, real-time interactive prompting of our model.

Losses and training. We supervise mask prediction with
the linear combination of focal loss [65] and dice loss [73]
used in [14]. We train for the promptable segmentation task
using a mixture of geometric prompts (for text prompts see
§7.5). Following [92, 37], we simulate an interactive setup
by randomly sampling prompts in 11 rounds per mask, al-
lowing SAM to integrate seamlessly into our data engine.

4. Segment Anything Data Engine

As segmentation masks are not abundant on the inter-
net, we built a data engine to enable the collection of our
1.1B mask dataset, SA-1B. The data engine has three
stages: (1) a model-assisted manual annotation stage, (2) a
semi-automatic stage with a mix of automatically predicted
masks and model-assisted annotation, and (3) a fully auto-
matic stage in which our model generates masks without
annotator input. We go into details of each next.

Assisted-manual stage. In the ﬁrst stage, resembling clas-
sic interactive segmentation, a team of professional annota-
tors labeled masks by clicking foreground / background ob-
ject points using a browser-based interactive segmentation
tool powered by SAM. Masks could be reﬁned using pixel-
precise “brush” and “eraser” tools. Our model-assisted an-
notation runs in real-time directly inside a browser (using
precomputed image embeddings) enabling a truly interac-
tive experience. We did not impose semantic constraints for
labeling objects, and annotators freely labeled both “stuff”
and “things” [1]. We suggested annotators label objects
they could name or describe, but did not collect these names
or descriptions. Annotators were asked to label objects in
order of prominence and were encouraged to proceed to the
next image once a mask took over 30 seconds to annotate.

5

,scorescorescore,,valid masksimageimageencoderimageembeddingmaskpointsboxtextprompt encodermask decoderconvAt the start of this stage, SAM was trained using com-
mon public segmentation datasets. After sufﬁcient data an-
notation, SAM was retrained using only newly annotated
masks. As more masks were collected, the image encoder
was scaled from ViT-B to ViT-H and other architectural de-
tails evolved; in total we retrained our model 6 times. Av-
erage annotation time per mask decreased from 34 to 14
seconds as the model improved. We note that 14 seconds
is 6.5× faster than mask annotation for COCO [66] and
only 2× slower than bounding-box labeling with extreme
points [76, 71]. As SAM improved, the average number of
masks per image increased from 20 to 44 masks. Overall,
we collected 4.3M masks from 120k images in this stage.

Semi-automatic stage. In this stage, we aimed to increase
the diversity of masks in order to improve our model’s
ability to segment anything. To focus annotators on less
prominent objects, we ﬁrst automatically detected conﬁdent
masks. Then we presented annotators with images preﬁlled
with these masks and asked them to annotate any additional
unannotated objects. To detect conﬁdent masks, we trained
a bounding box detector [84] on all ﬁrst stage masks using a
generic “object” category. During this stage we collected an
additional 5.9M masks in 180k images (for a total of 10.2M
masks). As in the ﬁrst stage, we periodically retrained our
model on newly collected data (5 times). Average annota-
tion time per mask went back up to 34 seconds (excluding
the automatic masks) as these objects were more challeng-
ing to label. The average number of masks per image went
from 44 to 72 masks (including the automatic masks).

Fully automatic stage. In the ﬁnal stage, annotation was
fully automatic. This was feasible due to two major en-
hancements to our model. First, at the start of this stage, we
had collected enough masks to greatly improve the model,
including the diverse masks from the previous stage. Sec-
ond, by this stage we had developed the ambiguity-aware
model, which allowed us to predict valid masks even in am-
biguous cases. Speciﬁcally, we prompted the model with a
32×32 regular grid of points and for each point predicted
a set of masks that may correspond to valid objects. With
the ambiguity-aware model, if a point lies on a part or sub-
part, our model will return the subpart, part, and whole ob-
ject. The IoU prediction module of our model is used to se-
lect conﬁdent masks; moreover, we identiﬁed and selected
only stable masks (we consider a mask stable if threshold-
ing the probability map at 0.5 − δ and 0.5 + δ results in
similar masks). Finally, after selecting the conﬁdent and
stable masks, we applied non-maximal suppression (NMS)
to ﬁlter duplicates. To further improve the quality of smaller
masks, we also processed multiple overlapping zoomed-in
image crops. For further details of this stage, see §B. We
applied fully automatic mask generation to all 11M images
in our dataset, producing a total of 1.1B high-quality masks.
We describe and analyze the resulting dataset, SA-1B, next.

Figure 5: Image-size normalized mask center distributions.

5. Segment Anything Dataset

Our dataset, SA-1B, consists of 11M diverse, high-
resolution,
licensed, and privacy protecting images and
1.1B high-quality segmentation masks collected with our
data engine. We compare SA-1B with existing datasets
and analyze mask quality and properties. We are releasing
SA-1B to aid future development of foundation models for
computer vision. We note that SA-1B will be released un-
der a favorable license agreement for certain research uses
and with protections for researchers.

Images. We licensed a new set of 11M images from a
provider that works directly with photographers. These im-
ages are high resolution (3300×4950 pixels on average),
and the resulting data size can present accessibility and stor-
age challenges. Therefore, we are releasing downsampled
images with their shortest side set to 1500 pixels. Even af-
ter downsampling, our images are signiﬁcantly higher reso-
lution than many existing vision datasets (e.g., COCO [66]
images are ∼480×640 pixels). Note that most models today
operate on much lower resolution inputs. Faces and vehicle
license plates have been blurred in the released images.

Masks. Our data engine produced 1.1B masks, 99.1% of
which were generated fully automatically. Therefore, the
quality of the automatic masks is centrally important. We
compare them directly to professional annotations and look
at how various mask properties compare to prominent seg-
mentation datasets. Our main conclusion, as borne out in
the analysis below and the experiments in §7, is that our
automatic masks are high quality and effective for training
models. Motivated by these ﬁndings, SA-1B only includes
automatically generated masks.

Mask quality. To estimate mask quality, we randomly sam-
pled 500 images (∼50k masks) and asked our professional
annotators to improve the quality of all masks in these im-
ages. Annotators did so using our model and pixel-precise
“brush” and “eraser” editing tools. This procedure resulted
in pairs of automatically predicted and professionally cor-
rected masks. We computed IoU between each pair and
found that 94% of pairs have greater than 90% IoU (and
97% of pairs have greater than 75% IoU). For comparison,
prior work estimates inter-annotator consistency at 85-91%
IoU [44, 60]. Our experiments in §7 conﬁrm by human rat-
ings that mask quality is high relative to a variety of datasets
and that training our model on automatic masks is nearly as
good as using all masks produced by the data engine.

6

Figure 6: Dataset mask properties. The legend references the number of images and masks in each dataset. Note, that SA-1B
has 11× more images and 400× more masks than the largest existing segmentation dataset Open Images [60].

Figure 7: Estimated geographic distribution of SA-1B images. Most of the world’s countries have more than 1000 images in
SA-1B, and the three countries with the most images are from different parts of the world.

Mask properties. In Fig. 5 we plot the spatial distribution
of object centers in SA-1B compared to the largest existing
segmentation datasets. Common photographer biases are
present in all datasets. We observe that SA-1B has greater
coverage of image corners compared to LVIS v1 [44] and
ADE20K [117], the two most similarly distributed datasets,
while COCO [66] and Open Images V5 [60] have a more
prominent center bias. In Fig. 6 (legend) we compare these
datasets by size. SA-1B has 11× more images and 400×
more masks than the second largest, Open Images. On av-
erage, it has 36× more masks per image than Open Images.
The closest dataset in this respect, ADE20K, still has 3.5×
fewer masks per image. Fig. 6 (left) plots the masks-per-
image distribution. Next, we look at image-relative mask
size (square root of the mask area divided by image area)
in Fig. 6 (middle). As expected, since our dataset has more
masks per image, it also tends to include a greater percent-
age of small and medium relative-size masks. Finally, to
analyze shape complexity, we look at mask concavity (1
minus mask area divided by area of mask’s convex hull) in
Fig. 6 (right). Since shape complexity is correlated with
mask size, we control for the datasets’ mask size distribu-
tions by ﬁrst performing stratiﬁed sampling from binned
mask sizes. We observe that the concavity distribution of
our masks is broadly similar to that of other datasets.

6. Segment Anything RAI Analysis

We next perform a Responsible AI (RAI) analysis of our
work by investigating potential fairness concerns and bi-
ases when using SA-1B and SAM. We focus on the geo-
graphic and income distribution of SA-1B and fairness of
SAM across protected attributes of people. We also provide
dataset, data annotation, and model cards in §F.

% images
# countries #imgs #masks SA-1B COCO

SA-1B

Africa

Europe

Asia & Oceania

Latin America & Carib.

54
70
47
42
4
81
high income countries
middle income countries 108
28

low income countries

North America

O.I.
300k
28M 2.8% 3.0% 1.7%
3.9M 423M 36.2% 11.4% 14.3%
5.4M 540M 49.8% 34.2% 36.2%
36M 3.5% 3.1% 5.0%
380k
830k
80M 7.7% 48.3% 42.8%
5.8M 598M 54.0% 89.1% 87.5%
4.9M 499M 45.0% 10.5% 12.0%
9.4M 0.9% 0.4% 0.5%
100k

Table 1: Comparison of geographic and income representa-
tion. SA-1B has higher representation in Europe and Asia &
Oceania as well as middle income countries. Images from
Africa, Latin America & Caribbean, as well as low income
countries, are underrepresented in all datasets.

Geographic and income representation. We infer the
country images were photographed in using standard meth-
ods (see §C). In Fig. 7 we visualize the per-country image
counts in SA-1B (left) and the 50 countries with the most
images (right). We note that the top-three countries are
from different parts of the world. Next, in Table 1 we com-
pare the geographic and income representation of SA-1B,
COCO [66], and Open Images [60]. SA-1B has a substan-
tially higher percentage of images in Europe and Asia &
Oceania as well as in middle income countries. All datasets
underrepresent Africa as well as low income countries. We
note that in SA-1B, all regions, including Africa, have at
least 28 million masks, 10× more than the total number of
masks of any previous dataset. Finally, we observe that the
average number of masks per image (not shown) is fairly
consistent across region and income (94-108 per image).

7

SA-1B11M images1129M (1.1B) masksLVIS v10.120M images1.5M masksCOCO0.123M images0.9M masksADE20K0.028M images0.7M masksOpen Images1M images2.7M masks<1011-5051-100101-200>200Number of masks per image04080Percent of images0.000.250.500.75Relative segmentation mask size10010−2Percent of masks0.00.20.40.60.8Concavity051015Percent of masksPer countryimage count≥ 100k< 100k< 10k< 1kRUSTHAUSAITAGBRDEUESPIDNUKRFRAJPNMYSTURINDCHNPOLNLDVNMBRACANGRCAUSPRTCZEBLRROUKORAREAUTSWETWNHKGCHEISRSGPHUNBELHRVBGRPHLKAZMEXNORMMRZAFSRBDNKMARFINLVA50 most common countries (ISO codes)0200k400k600k800kNumber of images per countryAsia & OceaniaAfricaEuropeNorth AmericaLatin America & CaribbeanmIoU at

1 point

3 points

54.4 ±1.7 90.4 ±0.6
55.7 ±1.7 90.1 ±0.6

perceived gender presentation
feminine
masculine
perceived age group
older
middle
young

62.9 ±6.7 92.6 ±1.3
54.5 ±1.3 90.2 ±0.5
54.2 ±2.2 91.2 ±0.7

mIoU at

3 points

1 point
perceived skin tone
1 52.9 ±2.2 91.0 ±0.9
2 51.5 ±1.4 91.1 ±0.5
3 52.2 ±1.9 91.4 ±0.7
4 51.5 ±2.7 91.7 ±1.0
5 52.4 ±4.2 92.5 ±1.4
6 56.7 ±6.3 91.2 ±2.4

Table 2: SAM’s performance segmenting people across per-
ceived gender presentation, age group, and skin tone. 95%
conﬁdence intervals are shown. Within each grouping, all
conﬁdence intervals overlap except older vs. middle.

Fairness in segmenting people. We investigate potential
fairness concerns across perceived gender presentation, per-
ceived age group, and perceived skin tone by measuring
the performance discrepancy of SAM between groups. We
use the More Inclusive Annotations for People (MIAP) [87]
dataset for gender presentation and age and a proprietary
dataset for skin tone (see §C). Our evaluation uses simu-
lated interactive segmentation with random sampling of 1
and 3 points (see §D). Table 2 (top left) shows results for
perceived gender presentation. We note that females have
been shown to be underrepresented in detection and seg-
mentation datasets [115], but observe that SAM performs
similarly across groups. We repeat the analysis for per-
ceived age in Table 2 (bottom left), noting that those who
are perceived to be younger and older have been shown to
be underrepresented in large-scale datasets [110]. SAM per-
forms best on those who are perceived older (although the
conﬁdence interval is large). Finally, we repeat the anal-
ysis for perceived skin tone in Table 2 (right), noting that
those with lighter apparent skin tones have been shown to
be overrepresented and those with darker skin tones under-
represented in large-scale datasets [110]. As MIAP does
not contain perceived skin tone annotations, we use a pro-
prietary dataset that contains annotations for the perceived
Fitzpatrick skin type [36], which ranges from 1 (lightest
skin tone) to 6 (darkest skin tone). While the means vary
somewhat, we do not ﬁnd a signiﬁcant difference across
groups. We believe our ﬁndings stem from the nature of
the task, and acknowledge biases may arise when SAM is
used as a component in larger systems. Finally, in §C we
extend the analysis to segmenting clothing where we ﬁnd
an indication of bias across perceived gender presentation.

7. Zero-Shot Transfer Experiments

In this section, we present zero-shot transfer experiments
with SAM, the Segment Anything Model. We consider ﬁve
tasks, four of which differ signiﬁcantly from the promptable
segmentation task used to train SAM. These experiments
evaluate SAM on datasets and tasks that were not seen dur-

8

ing training (our usage of “zero-shot transfer” follows its
usage in CLIP [82]). The datasets may include novel image
distributions, such as underwater or ego-centric images (e.g.
Fig. 8) that, to our knowledge, do not appear in SA-1B.

Our experiments begin by testing the core goal of
promptable segmentation: producing a valid mask from any
prompt. We emphasize the challenging scenario of a single
foreground point prompt, since it is more likely to be am-
biguous than other more speciﬁc prompts. Next, we present
a sequence of experiments that traverse low, mid, and high-
level image understanding and roughly parallel the histori-
cal development of the ﬁeld. Speciﬁcally, we prompt SAM
to (1) perform edge detection, (2) segment everything, i.e.
object proposal generation, (3) segment detected objects,
i.e. instance segmentation, and (4), as a proof-of-concept, to
segment objects from free-form text. These four tasks dif-
fer signiﬁcantly from the promptable segmentation task that
SAM was trained on and are implemented via prompt engi-
neering. Our experiments conclude with an ablation study.

Implementation. Unless otherwise speciﬁed:
(1) SAM
uses an MAE [47] pre-trained ViT-H [33] image encoder
and (2) SAM was trained on SA-1B, noting that this dataset
includes only automatically generated masks from the ﬁnal
stage of our data engine. For all other model and training
details, such as hyperparameters, refer to §A.

7.1. Zero-Shot Single Point Valid Mask Evaluation

Task. We evaluate segmenting an object from a single fore-
ground point. This task is ill-posed as one point can refer
to multiple objects. Ground truth masks in most datasets
do not enumerate all possible masks, which can make au-
tomatic metrics unreliable. Therefore, we supplement the
standard mIoU metric (i.e., the mean of all IoUs between
predicted and ground truth masks) with a human study in
which annotators rate mask quality from 1 (nonsense) to 10
(pixel-perfect). See §D.1, §E, and §G for additional details.
By default, we sample points from the “center” of ground
truth masks (at a maximal value of the mask’s interior dis-
tance transform), following the standard evaluation proto-
col in interactive segmentation [92]. Since SAM is capable
of predicting multiple masks, we evaluate only the model’s
most conﬁdent mask by default. The baselines are all
single-mask methods. We compare mainly to RITM [92],
a strong interactive segmenter that performs best on our
benchmark compared to other strong baselines [67, 18].

Datasets. We use a newly compiled suite of 23 datasets
with diverse image distributions. Fig. 8 lists the datasets
and shows a sample from each one (see appendix Table 7 for
more details). We use all 23 datasets for mIoU evaluation.
For the human study, we use the subset listed in Fig. 9b
(due to the resource requirements of such studies). This
subset includes both datasets for which SAM outperforms
and underperforms RITM according to automatic metrics.

ADE20K [117]

BBBC038v1 [12]

Cityscapes [25]

DOORS [80]

DRAM [24]

EgoHOS [113]

GTEA [34, 63]

Hypersim [86]

IBD [17]

iShape [111]

LVIS [44]

NDD20 [100]

NDISPark [22, 23]

OVIS [81]

PPDLS [74]

Plittersdorf [46]

STREETS [91]

TimberSeg [38]

TrashCan [52]

VISOR [28, 27]

WoodScape [112]

PIDRay [104]

ZeroWaste-f [6]

Figure 8: Samples from the 23 diverse segmentation datasets used to evaluate SAM’s zero-shot transfer capabilities.

PPDLS [74]
BBBC038v1 [12]
DOORS [80]
TimberSeg [38]
NDD20 [100]
LVIS [44]
STREETS [91]
ZeroWaste-f [6]
iShape [111]
ADE20K [117]
OVIS [81]
Hypersim [86]
NDISPark [22, 23]
VISOR [28, 27]
Plittersdorf [46]
EgoHOS [113]
IBD [17]
WoodScape [112]
Cityscapes [25]
PIDRay [104]
DRAM [24]
TrashCan [52]
GTEA [34, 63]

-21.4

-15.0

-20

+46.9

+44.7

+41.1

+28.9

+21.1

+18.5
+17.3

+9.1
+8.8
+7.8
+7.0
+6.1

+2.7
+1.8
+1.5
+0.8

(b) Mask quality ratings by human annotators

-0.3
-0.6

-2.0

-5.8
-6.5

0
IoU delta at 1 center point

+20

+40

(a) SAM vs. RITM [92] on 23 datasets

(c) Center points (default)

(d) Random points

Figure 9: Point to mask evaluation on 23 datasets. (a) Mean IoU of SAM and the strongest single point segmenter, RITM [92].
Due to ambiguity, a single mask may not match ground truth; circles show “oracle” results of the most relevant of SAM’s 3
predictions. (b) Per-dataset comparison of mask quality ratings by annotators from 1 (worst) to 10 (best). All methods use
the ground truth mask center as the prompt. (c, d) mIoU with varying number of points. SAM signiﬁcantly outperforms prior
interactive segmenters with 1 point and is on par with more points. Low absolute mIoU at 1 point is the result of ambiguity.

Results. First, we look at automatic evaluation on the full
suite of 23 datasets using mIoU. We compare per-dataset
results in Fig. 9a against RITM. SAM yields higher re-
sults on 16 of the 23 datasets, by as much as ∼47 IoU. We
also present an “oracle” result, in which the most relevant
of SAM’s 3 masks is selected by comparing them to the
ground truth, rather than selecting the most conﬁdent mask.
This reveals the impact of ambiguity on automatic evalu-
ation. In particular, with the oracle to perform ambiguity
resolution, SAM outperforms RITM on all datasets.

Results of the human study are presented in Fig. 9b. Er-
ror bars are 95% conﬁdence intervals for mean mask rat-
ings (all differences are signiﬁcant; see §E for details). We
observe that the annotators consistently rate the quality of
SAM’s masks substantially higher than the strongest base-
line, RITM. An ablated, “ambiguity-unaware” version of
SAM with a single output mask has consistently lower rat-
ings, though still higher than RITM. SAM’s mean ratings

fall between 7 and 9, which corresponds to the qualitative
rating guideline: “A high score (7-9): The object is identi-
ﬁable and errors are small and rare (e.g., missing a small,
heavily obscured disconnected component, ...).” These re-
sults indicate that SAM has learned to segment valid masks
from a single point. Note that for datasets like DRAM and
IBD, where SAM is worse on automatic metrics, it receives
consistently higher ratings in the human study.

Fig. 9c shows additional baselines, SimpleClick [67] and
FocalClick [18], which obtain lower single point perfor-
mance than RITM and SAM. As the number of points in-
creases from 1 to 9, we observe that the gap between meth-
ods decreases. This is expected as the task becomes easier;
also, SAM is not optimized for the very high IoU regime.
Finally, in Fig. 9d we replace the default center point sam-
pling with random point sampling. We observe that the gap
between SAM and the baselines grows and SAM is able to
achieve comparable results under either sampling method.

9

LVISVISORDRAMIBDNDD20OVISiShapeDatasets579Avg. mask ratingGround TruthSAMSAM - single outputRITM12359Number of points5075mIoU (23 datasets)SAM (oracle)SAMRITMSimpleClickFocalClick12359Number of points5075mIoU (23 datasets)SAM (oracle)image

ground truth

SAM

Figure 10: Zero-shot edge prediction on BSDS500. SAM
was not trained to predict edge maps nor did it have access
to BSDS images or annotations during training.

year
2015
2022

method
HED [108]
EDETR [79]
zero-shot transfer methods:
Sobel ﬁlter
Canny [13]
Felz-Hutt [35]
SAM

1968
1986
2004
2023

ODS
.788
.840

.539
.600
.610
.768

OIS
.808
.858

-
.640
.640
.786

AP
.840
.896

-
.580
.560
.794

R50
.923
.930

-
-
-
.928

Table 3: Zero-shot transfer to edge detection on BSDS500.

7.2. Zero-Shot Edge Detection

Approach. We evaluate SAM on the classic low-level task
of edge detection using BSDS500 [72, 3]. We use a sim-
pliﬁed version of our automatic mask generation pipeline.
Speciﬁcally, we prompt SAM with a 16×16 regular grid of
foreground points resulting in 768 predicted masks (3 per
point). Redundant masks are removed by NMS. Then, edge
maps are computed using Sobel ﬁltering of unthresholded
mask probability maps and standard lightweight postpro-
cessing, including edge NMS (see §D.2 for details).

Results. We visualize representative edge maps in Fig. 10
(see Fig. 15 for more). Qualitatively, we observe that even
though SAM was not trained for edge detection, it produces
reasonable edge maps. Compared to the ground truth, SAM
predicts more edges, including sensible ones that are not an-
notated in BSDS500. This bias is reﬂected quantitatively in
Table 3: recall at 50% precision (R50) is high, at the cost of
precision. SAM naturally lags behind state-of-the-art meth-
ods that learn the biases of BSDS500, i.e., which edges to
suppress. Nevertheless, SAM performs well compared to
pioneering deep learning methods such as HED [108] (also
trained on BSDS500) and signiﬁcantly better than prior,
though admittedly outdated, zero-shot transfer methods.

7.3. Zero-Shot Object Proposals

Approach. Next, we evaluate SAM on the mid-level task
of object proposal generation [2, 102]. This task has played
an important role in object detection research, serving as an

10

all
63.0

method
ViTDet-H [62]
zero-shot transfer methods:
SAM – single out.
SAM

54.9
59.3

mask AR@1000
large
87.0

freq.
63.1

small med.
80.8
51.7

com.
63.3

rare
58.3

42.8
45.5

76.7
81.6

74.4
86.9

54.7
59.1

59.8
63.9

62.0
65.8

Table 4: Object proposal generation on LVIS v1. SAM is
applied zero-shot, i.e. it was not trained for object proposal
generation nor did it access LVIS images or annotations.

intermediate step in pioneering systems (e.g., [102, 41, 84]).
To generate object proposals, we run a slightly modiﬁed
version of our automatic mask generation pipeline and out-
put the masks as proposals (see §D.3 for details).

We compute the standard average recall (AR) metric on
LVIS v1 [44]. We focus on LVIS because its large number
of categories presents a challenging test. We compare to
a strong baseline implemented as a ViTDet [62] detector
(with cascade Mask R-CNN [48, 11] ViT-H). We note that
this “baseline” corresponds to the “Detector Masquerading
as Proposal generator” (DMP) method [16] that was shown
to game AR, making it a truly demanding comparison.

Results.
In Table 4 we see unsurprisingly that using the
detections from ViTDet-H as object proposals (i.e.,
the
DMP method [16] that games AR) performs the best over-
all. However, SAM does remarkably well on several met-
rics. Notably, it outperforms ViTDet-H on medium and
large objects, as well as rare and common objects. In fact,
SAM only underperforms ViTDet-H on small objects and
frequent objects, where ViTDet-H can easily learn LVIS-
speciﬁc annotation biases since it was trained on LVIS, un-
like SAM. We also compare against an ablated ambiguity-
unaware version of SAM (“single out.”), which performs
signiﬁcantly worse than SAM on all AR metrics.

7.4. Zero-Shot Instance Segmentation

Approach. Moving to higher-level vision, we use SAM
as the segmentation module of an instance segmenter. The
implementation is simple: we run a object detector (the
ViTDet used before) and prompt SAM with its output
boxes. This illustrates composing SAM in a larger system.

Results. We compare the masks predicted by SAM and
ViTDet on COCO and LVIS in Table 5. Looking at the
mask AP metric we observe gaps on both datasets, where
SAM is reasonably close, though certainly behind ViTDet.
By visualizing outputs, we observed that SAM masks are
often qualitatively better than those of ViTDet, with crisper
boundaries (see §D.4 and Fig. 16). To investigate this ob-
servation, we conducted an additional human study asking
annotators to rate the ViTDet masks and SAM masks on the
1 to 10 quality scale used before. In Fig. 11 we observe that
SAM consistently outperforms ViTDet in the human study.

COCO [66]
AP APS APM APL
68.9

method
ViTDet-H [62] 51.0
zero-shot transfer methods (segmentation module only):
SAM

32.0

54.3

51.0

30.8

61.7

46.5

44.7

LVIS v1 [44]
AP APS APM APL
58.0 66.3
46.6

35.0

32.5

57.6 65.5

Table 5: Instance segmentation results. SAM is prompted
with ViTDet boxes to do zero-shot segmentation. The fully-
supervised ViTDet outperforms SAM, but the gap shrinks
on the higher-quality LVIS masks. Interestingly, SAM out-
performs ViTDet according to human ratings (see Fig. 11).

(cid:51)

(cid:55)

(cid:55)

(cid:51)

(cid:51)

(cid:51)

Figure 11: Mask quality rating distribution from our human
study for ViTDet and SAM, both applied to LVIS ground
truth boxes. We also report LVIS and COCO ground truth
quality. The legend shows rating means and 95% conﬁ-
dence intervals. Despite its lower AP (Table 5), SAM has
higher ratings than ViTDet, suggesting that ViTDet exploits
biases in the COCO and LVIS training data.

We hypothesize that on COCO, where the mask AP gap
is larger and the ground truth quality is relatively low (as
borne out by the human study), ViTDet learns the speciﬁc
biases of COCO masks. SAM, being a zero-shot method,
is unable to exploit these (generally undesirable) biases.
The LVIS dataset has higher quality ground truth, but there
are still speciﬁc idiosyncrasies (e.g., masks do not contain
holes, they are simple polygons by construction) and biases
for modal vs. amodal masks. Again, SAM is not trained to
learn these biases, while ViTDet can exploit them.

7.5. Zero-Shot Text-to-Mask

Approach. Finally, we consider an even higher-level task:
segmenting objects from free-form text. This experiment
is a proof-of-concept of SAM’s ability to process text
prompts. While we used the exact same SAM in all prior
experiments, for this one SAM’s training procedure is mod-
iﬁed to make it text-aware, but in a way that does not require
new text annotations. Speciﬁcally, for each manually col-
lected mask with area larger than 1002 we extract the CLIP
image embedding. Then, during training, we prompt SAM
with the extracted CLIP image embeddings as its ﬁrst in-
teraction. The key observation here is that because CLIP’s
image embeddings are trained to align with its text embed-
dings, we can train with image embeddings, but use text
embeddings for inference. That is, at inference time we run
text through CLIP’s text encoder and then give the resulting
text embedding as a prompt to SAM (see §D.5 for details).

11

Figure 12: Zero-shot text-to-mask. SAM can work with
simple and nuanced text prompts. When SAM fails to make
a correct prediction, an additional point prompt can help.

Results. We show qualitative results in Fig. 12. SAM
can segment objects based on simple text prompts like “a
wheel” as well as phrases like “beaver tooth grille”. When
SAM fails to pick the right object from a text prompt only,
an additional point often ﬁxes the prediction, similar to [31].

7.6. Ablations

We perform several ablations on our 23 dataset suite with
the single center point prompt protocol. Recall that a sin-
gle point may be ambiguous and that ambiguity may not
be represented in the ground truth, which contains only a
single mask per point. Since SAM is operating in a zero-
shot transfer setting there can be systematic biases between
SAM’s top-ranked mask vs. the masks resulting from data
annotation guidelines. We therefore additionally report the
best mask with respect to the ground truth (“oracle”).

Fig. 13 (left) plots SAM’s performance when trained on
cumulative data from the data engine stages. We observe
that each stage increases mIoU. When training with all three
stages, the automatic masks vastly outnumber the manual
and semi-automatic masks. To address this, we found that
oversampling the manual and semi-automatic masks during
training by 10× gave best results. This setup complicates
training. We therefore tested a fourth setup that uses only
the automatically generated masks. With this data, SAM
performs only marginally lower than using all data (∼0.5
mIoU). Therefore, by default we use only the automatically
generated masks to simplify the training setup.

In Fig. 13 (middle) we look at the impact of data volume.
The full SA-1B contains 11M images, which we uniformly
subsample to 1M and 0.1M for this ablation. At 0.1M im-
ages, we observe a large mIoU decline under all settings.
However, with 1M images, about 10% of the full dataset,
we observe results comparable to using the full dataset.
This data regime, which still includes approximately 100M
masks, may be a practical setting for many use cases.

12345678910Mask quality rating02040Percent of ratings8.6 ± 0.06, LVIS GT8.1 ± 0.07, SAM7.9 ± 0.08, ViTDet-H7.6 ± 0.12, COCO GT     “a wheel”     “beaver tooth grille”     “a wiper”     “a wiper” + point     “wipers”     “wipers” + pointFigure 13: Ablation studies of our data engine stages, image encoder scaling, and training data scaling. (Left) Each data
engine stage leads to improvements on our 23 dataset suite, and training with only the automatic data (our default) yields
similar results to using data from all three stages. (Middle) SAM trained with ∼10% of SA-1B and full SA-1B is comparable.
We train with all 11M images by default, but using 1M images is a reasonable practical setting. (Right) Scaling SAM’s image
encoder shows meaningful, yet saturating gains. Nevertheless, smaller image encoders may be preferred in certain settings.

Finally, Fig. 13 (right) shows results with ViT-B, ViT-L,
and ViT-H image encoders. ViT-H improves substantially
over ViT-B, but has only marginal gains over ViT-L. Further
image encoder scaling does not appear fruitful at this time.

8. Discussion

Foundation models. Pre-trained models have been adapted
to downstream tasks since the early days of machine learn-
ing [99]. This paradigm has become increasingly impor-
tant in recent years with a growing emphasis on scale, and
such models have recently been (re-)branded as “founda-
i.e. models that are “trained on broad data
tion models”:
at scale and are adaptable to a wide range of downstream
tasks” [8]. Our work correlates well with this deﬁnition,
though we note that a foundation model for image segmen-
tation is an inherently limited scope, since it represents an
important, yet fractional, subset of computer vision. We
also contrast one aspect of our approach with [8], which
emphasizes the role of self-supervised learning in founda-
tion models. While our model is initialized with a self-
supervised technique (MAE [47]), the vast majority of its
capabilities come from large-scale supervised training. In
cases where data engines can scale available annotations,
like ours, supervised training provides an effective solution.

Compositionality. Pre-trained models can power new ca-
pabilities even beyond ones imagined at the moment of
training. One prominent example is how CLIP [82] is used
as a component in larger systems, such as DALL·E [83].
Our goal is to make this kind of composition straightfor-
ward with SAM. We aim to achieve this by requiring SAM
to predict a valid mask for a wide range of segmentation
prompts. The effect is to create a reliable interface between
SAM and other components. For example, MCC [106] can
easily use SAM to segment an object of interest and achieve
strong generalization to unseen objects for 3D reconstruc-
tion from a single RGB-D image. In another example, SAM
can be prompted with gaze points detected by a wearable
device, enabling new applications. Thanks to SAM’s abil-
ity to generalize to new domains like ego-centric images,
such systems work without need for additional training.

Limitations. While SAM performs well in general, it is
not perfect. It can miss ﬁne structures, hallucinates small
disconnected components at times, and does not produce
boundaries as crisply as more computationally intensive
methods that “zoom-in”, e.g. [18]. In general, we expect
dedicated interactive segmentation methods to outperform
SAM when many points are provided, e.g. [67]. Unlike
these methods, SAM is designed for generality and breadth
of use rather than high IoU interactive segmentation. More-
over, SAM can process prompts in real-time, but neverthe-
less SAM’s overall performance is not real-time when using
a heavy image encoder. Our foray into the text-to-mask task
is exploratory and not entirely robust, although we believe
it can be improved with more effort. While SAM can per-
form many tasks, it is unclear how to design simple prompts
that implement semantic and panoptic segmentation. Fi-
nally, there are domain-speciﬁc tools, such as [7], that we
expect to outperform SAM in their respective domains.

Conclusion. The Segment Anything project is an attempt to
lift image segmentation into the era of foundation models.
Our principal contributions are a new task (promptable seg-
mentation), model (SAM), and dataset (SA-1B) that make
this leap possible. Whether SAM achieves the status of a
foundation model remains to be seen by how it is used in
the community, but regardless we expect the perspective of
this work, the release of over 1B masks, and our promptable
segmentation model will help pave the path ahead.

Acknowledgments. We would like to thank Aaron Ad-
cock and Jitendra Malik for helpful discussion. We thank
Vaibhav Aggarwal and Yanghao Li for help with scal-
ing the model. We thank Cheng-Yang Fu, Jiabo Hu, and
Robert Kuo for help with data annotation platform. We
thank Allen Goodman and Bram Wasti for help in optimiz-
ing web-version of our model. Finally, we thank Morteza
Behrooz, Ashley Gabriel, Ahuva Goldstand, Sumanth Gur-
ram, Somya Jain, Devansh Kukreja, Joshua Lane, Lilian
Luong, Mallika Malhotra, William Ngan, Omkar Parkhi,
Nikhil Raina, Dirk Rowe, Neil Sejoor, Vanessa Stark, Bala
Varadarajan, and Zachary Winstrom for their help in mak-
ing the demo, dataset viewer, and other assets and tooling.

12

manual+ semiautomatic+ automaticautomaticonlyTraining data stages506070mIoU (23 datasets)1 point (oracle)1 point0.1M1M11MTraining images707580mIoU (23 datasets)1 point (oracle)2 points3 points5 points91MViT-B308MViT-L636MViT-HNumber of parameters606570mIoU (23 datasets)1 point (oracle)1 pointReferences

[1] Edward H Adelson. On seeing stuff: the perception of materials by
humans and machines. Human vision and electronic imaging VI,
2001. 5

[2] Bogdan Alexe, Thomas Deselaers, and Vittorio Ferrari. What is an

object? CVPR, 2010. 4, 10

[3] Pablo Arbel´aez, Michael Maire, Charless Fowlkes, and Jitendra
Malik. Contour detection and hierarchical image segmentation.
TPAMI, 2010. 4, 10, 21, 28

[4] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer

normalization. arXiv:1607.06450, 2016. 16

[5] Hangbo Bao, Li Dong, and Furu Wei. BEiT: BERT pre-training of

image transformers. arXiv:2106.08254, 2021. 17

[6] Dina Bashkirova, Mohamed Abdelfattah, Ziliang Zhu, James Akl,
Fadi Alladkani, Ping Hu, Vitaly Ablavsky, Berk Calli, Sarah Adel
Bargal, and Kate Saenko. ZeroWaste dataset: Towards deformable
object segmentation in cluttered scenes. CVPR, 2022. 9, 20

[7] Stuart Berg, Dominik Kutra, Thorben Kroeger, Christoph N.
Straehle, Bernhard X. Kausler, Carsten Haubold, Martin Schiegg,
Janez Ales, Thorsten Beier, Markus Rudy, Kemal Eren, Jaime I.
Cervantes, Buote Xu, Fynn Beuttenmueller, Adrian Wolny, Chong
Zhang, Ullrich Koethe, Fred A. Hamprecht, and Anna Kreshuk.
ilastik: interactive machine learning for (bio)image analysis. Na-
ture Methods, 2019. 12

[8] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman,
Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette
Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportu-
nities and risks of foundation models. arXiv:2108.07258, 2021. 1,
12

[9] Gustav Bredell, Christine Tanner, and Ender Konukoglu. Iterative
interaction training for segmentation editing networks. MICCAI,
2018. 17

[10] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah,
Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav
Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel
Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child,
Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris
Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Ben-
jamin Chess, Jack Clark, Christopher Berner, Sam McCandlish,
Alec Radford, Ilya Sutskever, and Dario Amodei. Language models
are few-shot learners. NeurIPS, 2020. 1, 4

[11] Zhaowei Cai and Nuno Vasconcelos. Cascade R-CNN: Delving into

high quality object detection. CVPR, 2018. 10

[12] Juan C. Caicedo, Allen Goodman, Kyle W. Karhohs, Beth A. Ci-
mini, Jeanelle Ackerman, Marzieh Haghighi, CherKeng Heng, Tim
Becker, Minh Doan, Claire McQuin, Mohammad Rohban, Shan-
tanu Singh, and Anne E. Carpenter. Nucleus segmentation across
imaging experiments: the 2018 data science bowl. Nature Methods,
2019. 9, 19, 20

[13] John Canny. A computational approach to edge detection. TPAMI,

1986. 10, 21

[14] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas
Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end
object detection with Transformers. ECCV, 2020. 5, 16, 17
[15] Guillaume Charpiat, Matthias Hofmann, and Bernhard Sch¨olkopf.
Automatic image colorization via multimodal predictions. ECCV,
2008. 5, 17

[16] Neelima Chavali, Harsh Agrawal, Aroma Mahendru, and Dhruv
Batra. Object-proposal evaluation protocol is’ gameable’. CVPR,
2016. 10, 21

[17] Jiazhou Chen, Yanghui Xu, Shufang Lu, Ronghua Liang, and Lian-
IEEE
gliang Nan. 3D instance segmentation of MVS buildings.
Transactions on Geoscience and Remote Sensing, 2022. 9, 19, 20,
23, 24

[18] Xi Chen, Zhiyan Zhao, Yilei Zhang, Manni Duan, Donglian Qi, and
Hengshuang Zhao. FocalClick: towards practical interactive image
segmentation. CVPR, 2022. 8, 9, 12, 19

[19] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kir-
illov, and Rohit Girdhar. Masked-attention mask transformer for
universal image segmentation. CVPR, 2022. 4

[20] Bowen Cheng, Alex Schwing, and Alexander Kirillov.

Per-
pixel classiﬁcation is not all you need for semantic segmentation.
NeurIPS, 2021. 5, 16, 17

[21] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten
Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won
Chung, Charles Sutton, Sebastian Gehrmann, et al. PaLM: Scaling
language modeling with pathways. arXiv:2204.02311, 2022. 1
[22] Luca Ciampi, Carlos Santiago, Joao Costeira, Claudio Gennaro, and
Giuseppe Amato. Domain adaptation for trafﬁc density estimation.
International Joint Conference on Computer Vision, Imaging and
Computer Graphics Theory and Applications, 2021. 9, 20

[23] Luca Ciampi, Carlos Santiago, Joao Costeira, Claudio Gennaro, and
Giuseppe Amato. Night and day instance segmented park (NDIS-
Park) dataset: a collection of images taken by day and by night for
vehicle detection, segmentation and counting in parking areas. Zen-
odo, 2022. 9, 20

[24] Nadav Cohen, Yael Newman, and Ariel Shamir. Semantic segmen-
tation in art paintings. Computer Graphics Forum, 2022. 9, 19, 20,
23, 24

[25] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld,
Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth,
and Bernt Schiele. The Cityscapes dataset for semantic urban scene
understanding. CVPR, 2016. 9, 19, 20

[26] Bruno da Silva, George Konidaris, and Andrew Barto. Learning

parameterized skills. ICML, 2012. 4

[27] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino
Furnari, Jian Ma, Evangelos Kazakos, Davide Moltisanti, Jonathan
Munro, Toby Perrett, Will Price, and Michael Wray. Rescaling
egocentric vision: Collection, pipeline and challenges for EPIC-
KITCHENS-100. IJCV, 2022. 9, 20, 23, 24

[28] Ahmad Darkhalil, Dandan Shan, Bin Zhu, Jian Ma, Amlan Kar,
Richard Higgins, Sanja Fidler, David Fouhey, and Dima Damen.
EPIC-KITCHENS VISOR benchmark: Video segmentations and
object relations. NeurIPS, 2022. 9, 19, 20, 23, 24

[29] Terrance De Vries, Ishan Misra, Changhan Wang, and Laurens
Van der Maaten. Does object recognition work for everyone? CVPR
workshops, 2019. 18

[30] Mark D´ıaz, Ian Kivlichan, Rachel Rosen, Dylan Baker, Razvan
Amironesei, Vinodkumar Prabhakaran, and Emily Denton. Crowd-
WorkSheets: Accounting for individual and collective identities un-
derlying crowdsourced dataset annotation. ACM Conference on
Fairness, Accountability, and Transparency, 2022. 25

[31] Henghui Ding, Scott Cohen, Brian Price, and Xudong Jiang.
PhraseClick: toward achieving ﬂexible interactive segmentation by
phrase and click. ECCV, 2020. 11

[32] Piotr Doll´ar and C Lawrence Zitnick. Fast edge detection using

structured forests. TPAMI, 2014. 21

[33] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk
Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa De-
hghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob
Uszkoreit, and Neil Houlsby. An image is worth 16x16 words:
ICLR, 2021. 5, 8,
Transformers for image recognition at scale.
16

[34] Alireza Fathi, Xiaofeng Ren, and James M. Rehg. Learning to rec-
ognize objects in egocentric activities. CVPR, 2011. 9, 19, 20
[35] Pedro F Felzenszwalb and Daniel P Huttenlocher. Efﬁcient graph-

based image segmentation. IJCV, 2004. 10

[36] Thomas B. Fitzpatrick. The validity and practicality of sun-reactive

skin types i through vi. Archives of Dermatology, 1988. 8

[37] Marco Forte, Brian Price, Scott Cohen, Ning Xu, and Franc¸ois
Getting to 99% accuracy in interactive segmentation.

Piti´e.
arXiv:2003.07932, 2020. 5, 17

[38] Jean-Michel Fortin, Olivier Gamache, Vincent Grondin, Franc¸ois
Instance segmentation for au-

Pomerleau, and Philippe Gigu`ere.
tonomous log grasping in forestry operations. IROS, 2022. 9, 20

13

[39] Timnit Gebru,

Jamie Morgenstern, Briana Vecchione,

Jen-
nifer Wortman Vaughan, Hanna Wallach, Hal Daum´e Iii, and Kate
Crawford. Datasheets for datasets. Communications of the ACM,
2021. 25

[40] Golnaz Ghiasi, Yin Cui, Aravind Srinivas, Rui Qian, Tsung-Yi Lin,
Ekin D Cubuk, Quoc V Le, and Barret Zoph. Simple copy-paste is a
strong data augmentation method for instance segmentation. CVPR,
2021. 16, 18, 22

[41] Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik.
Rich feature hierarchies for accurate object detection and semantic
segmentation. CVPR, 2014. 10

[42] Priya Goyal, Piotr Doll´ar, Ross Girshick, Pieter Noordhuis, Lukasz
Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and
Kaiming He. Accurate, large minibatch SGD: Training ImageNet
in 1 hour. arXiv:1706.02677, 2017. 17

[43] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary
Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger,
Hao Jiang, Miao Liu, Xingyu Liu, Miguel Martin, Tushar Na-
garajan, Ilija Radosavovic, Santhosh Kumar Ramakrishnan, Fiona
Ryan, Jayant Sharma, Michael Wray, Mengmeng Xu, Eric Zhong-
cong Xu, Chen Zhao, Siddhant Bansal, Dhruv Batra, Vincent Car-
tillier, Sean Crane, Tien Do, Morrie Doulaty, Akshay Erapalli,
Christoph Feichtenhofer, Adriano Fragomeni, Qichen Fu, Chris-
tian Fuegen, Abrham Gebreselasie, Cristina Gonzalez, James Hillis,
Xuhua Huang, Yifei Huang, Wenqi Jia, Weslie Khoo, Jachym Ko-
lar, Satwik Kottur, Anurag Kumar, Federico Landini, Chao Li,
Yanghao Li, Zhenqiang Li, Karttikeya Mangalam, Raghava Mod-
hugu, Jonathan Munro, Tullie Murrell, Takumi Nishiyasu, Will
Price, Paola Ruiz Puentes, Merey Ramazanova, Leda Sari, Kiran
Somasundaram, Audrey Southerland, Yusuke Sugano, Ruijie Tao,
Minh Vo, Yuchen Wang, Xindi Wu, Takuma Yagi, Yunyi Zhu,
Pablo Arbelaez, David Crandall, Dima Damen, Giovanni Maria
Farinella, Bernard Ghanem, Vamsi Krishna Ithapu, C. V. Jawahar,
Hanbyul Joo, Kris Kitani, Haizhou Li, Richard Newcombe, Aude
Oliva, Hyun Soo Park, James M. Rehg, Yoichi Sato, Jianbo Shi,
Mike Zheng Shou, Antonio Torralba, Lorenzo Torresani, Mingfei
Yan, and Jitendra Malik. Ego4D: Around the World in 3,000 Hours
of Egocentric Video. CVPR, 2022. 20

[44] Agrim Gupta, Piotr Dollar, and Ross Girshick. LVIS: A dataset for
large vocabulary instance segmentation. CVPR, 2019. 2, 6, 7, 9, 10,
11, 19, 20, 21, 24

[45] Abner Guzman-Rivera, Dhruv Batra, and Pushmeet Kohli. Multiple
choice learning: Learning to produce multiple structured outputs.
NeurIPS, 2012. 5, 17

[46] Timm Haucke, Hjalmar S. K¨uhl,

and Volker Steinhage.
SOCRATES: Introducing depth in visual wildlife monitoring using
stereo vision. Sensors, 2022. 9, 20

[47] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar,
and Ross Girshick. Masked autoencoders are scalable vision learn-
ers. CVPR, 2022. 5, 8, 12, 16, 17

[48] Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross Girshick.

Mask R-CNN. ICCV, 2017. 10

[49] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep

residual learning for image recognition. CVPR, 2016. 16

[50] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units

(gelus). arXiv:1606.08415, 2016. 16

[51] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena
Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas,
Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training
compute-optimal large language models. arXiv:2203.15556, 2022.
1

[52] Jungseok Hong, Michael Fulton, and Junaed Sattar. TrashCan: A
semantically-segmented dataset towards visual detection of marine
debris. arXiv:2007.08097, 2020. 9, 19, 20

[53] Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Wein-
berger. Deep networks with stochastic depth. ECCV, 2016. 17
[54] Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita Orlov,
and Humphrey Shi. Oneformer: One transformer to rule universal
image segmentation. arXiv:2211.06220, 2022. 4

14

[55] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh,
Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig.
Scaling up visual and vision-language representation learning with
noisy text supervision. ICML, 2021. 1

[56] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown,
Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey
Wu, and Dario Amodei. Scaling laws for neural language models.
arXiv:2001.08361, 2020. 1

[57] Michael Kass, Andrew Witkin, and Demetri Terzopoulos. Snakes:

Active contour models. IJCV, 1988. 4

[58] Dahun Kim, Tsung-Yi Lin, Anelia Angelova, In So Kweon, and
Weicheng Kuo. Learning open-world object proposals without
learning to classify. IEEE Robotics and Automation Letters, 2022.
21

[59] Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother,

and Piotr Doll´ar. Panoptic segmentation. CVPR, 2019. 4

[60] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan
Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo
Malloci, Alexander Kolesnikov, Tom Duerig, and Vittorio Ferrari.
The open images dataset v4: Uniﬁed image classiﬁcation, object
detection, and visual relationship detection at scale. IJCV, 2020. 2,
6, 7, 18, 19

[61] Alexandre Lacoste, Alexandra Luccioni, Victor Schmidt, and
Thomas Dandres. Quantifying the carbon emissions of machine
learning. arXiv:1910.09700, 2019. 28

[62] Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He. Explor-
ing plain vision transformer backbones for object detection. ECCV,
2022. 5, 10, 11, 16, 21, 23, 24

[63] Yin Li, Zhefan Ye, and James M. Rehg. Delving into egocentric

actions. CVPR, 2015. 9, 20

[64] Zhuwen Li, Qifeng Chen, and Vladlen Koltun. Interactive image

segmentation with latent diversity. CVPR, 2018. 5, 17, 19

[65] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr

Doll´ar. Focal loss for dense object detection. ICCV, 2017. 5, 17

[66] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro
Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Mi-
crosoft COCO: Common objects in context. ECCV, 2014. 2, 4, 6,
7, 11, 18, 19, 20

[67] Qin Liu, Zhenlin Xu, Gedas Bertasius, and Marc Niethammer. Sim-
pleClick: Interactive image segmentation with simple vision trans-
formers. arXiv:2210.11006, 2022. 8, 9, 12, 19

[68] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regu-

larization. ICLR, 2019. 17

[69] Cathy H Lucas, Daniel OB Jones, Catherine J Hollyhead, Robert H
Condon, Carlos M Duarte, William M Graham, Kelly L Robinson,
Kylie A Pitt, Mark Schildhauer, and Jim Regetz. Gelatinous zoo-
plankton biomass in the global oceans: geographic variation and
environmental drivers. Global Ecology and Biogeography, 2014.
20

[70] Sabarinath Mahadevan, Paul Voigtlaender, and Bastian Leibe. Iter-
atively trained interactive segmentation. BMVC, 2018. 4, 17
[71] Kevis-Kokitsi Maninis, Sergi Caelles, Jordi Pont-Tuset, and Luc
Van Gool. Deep extreme cut: From extreme points to object seg-
mentation. CVPR, 2018. 6

[72] David Martin, Charless Fowlkes, Doron Tal, and Jitendra Malik.
A database of human segmented natural images and its applica-
tion to evaluating segmentation algorithms and measuring ecologi-
cal statistics. ICCV, 2001. 10, 21, 28

[73] Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-Net:
Fully convolutional neural networks for volumetric medical image
segmentation. 3DV, 2016. 5, 17

[74] Massimo Minervini, Andreas Fischbach, Hanno Scharr, and
Sotirios A. Tsaftaris. Finely-grained annotated datasets for image-
based plant phenotyping. Pattern Recognition Letters, 2016. 9, 20
[75] Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes,
Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Debo-
rah Raji, and Timnit Gebru. Model cards for model reporting. Pro-
ceedings of the conference on fairness, accountability, and trans-
parency, 2019. 25, 28

[76] Dim P Papadopoulos, Jasper RR Uijlings, Frank Keller, and Vittorio
ICCV,

Ferrari. Extreme clicking for efﬁcient object annotation.
2017. 6

[77] David Patterson, Joseph Gonzalez, Quoc Le, Chen Liang, Lluis-
Miquel Munguia, Daniel Rothchild, David So, Maud Texier, and
Jeff Dean. Carbon emissions and large neural network training.
arXiv:2104.10350, 2021. 28

[78] Matthew E Peters, Waleed Ammar, Chandra Bhagavatula, and Rus-
sell Power. Semi-supervised sequence tagging with bidirectional
language models. Proceedings of the 55th Annual Meeting of the
Association for Computational Linguistics, 2017. 18

[79] Mengyang Pu, Yaping Huang, Yuming Liu, Qingji Guan, and
Haibin Ling. EDTER: Edge detection with transformer. CVPR,
2022. 10

[80] Mattia Pugliatti and Francesco Topputo. DOORS: Dataset fOr

bOuldeRs Segmentation. Zenodo, 2022. 9, 20

[81] Jiyang Qi, Yan Gao, Yao Hu, Xinggang Wang, Xiaoyu Liu, Xiang
Bai, Serge Belongie, Alan Yuille, Philip Torr, and Song Bai. Oc-
cluded video instance segmentation: A benchmark. ICCV, 2022. 9,
20, 23, 24

[82] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh,
Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell,
Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. ICML, 2021. 1, 2, 4, 5,
8, 12, 16, 22

[83] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea
Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-
to-image generation. ICML, 2021. 1, 4, 12

[84] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster
R-CNN: Towards real-time object detection with region proposal
networks. NeurIPS, 2015. 6, 10

[85] Xiaofeng Ren and Jitendra Malik. Learning a classiﬁcation model

for segmentation. ICCV, 2003. 4

[86] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar,
Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M.
Susskind. Hypersim: A photorealistic synthetic dataset for holistic
indoor scene understanding. ICCV, 2021. 9, 19, 20

[87] Candice Schumann, Susanna Ricco, Utsav Prabhu, Vittorio Ferrari,
and Caroline Pantofaru. A step toward more inclusive people anno-
tations for fairness. Proceedings of the 2021 AAAI/ACM Conference
on AI, Ethics, and Society, 2021. 8, 19

[88] Seﬁk Ilkin Serengil and Alper Ozpinar. LightFace: A hybrid deep

face recognition framework. ASYU, 2020. 26

[89] Seﬁk Ilkin Serengil and Alper Ozpinar. HyperExtended LightFace:

A facial attribute analysis framework. ICEET, 2021. 26

[90] Jamie Shotton, John Winn, Carsten Rother, and Antonio Crimin-
isi. TextonBoost: Joint appearance, shape and context modeling for
mulit-class object recognition and segmentation. ECCV, 2006. 4

[91] Corey Snyder and Minh Do. STREETS: A novel camera network

dataset for trafﬁc ﬂow. NeurIPS, 2019. 9, 20

[92] Konstantin Soﬁiuk, Ilya A Petrov, and Anton Konushin. Reviving
iterative training with mask guidance for interactive segmentation.
ICIP, 2022. 5, 8, 9, 17, 19, 23, 24, 28

[93] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky,

Ilya
Sutskever, and Ruslan Salakhutdinov. Dropout: A simple way to
prevent neural networks from overﬁtting. The Journal of Machine
Learning Research, 2014. 16

[94] Chris Stauffer and W Eric L Grimson. Adaptive background mix-

ture models for real-time tracking. CVPR, 1999. 4

[95] Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara
Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ra-
mamoorthi, Jonathan Barron, and Ren Ng. Fourier features let net-
works learn high frequency functions in low dimensional domains.
NeurIPS, 2020. 5, 16

[96] Yansong Tang, Yi Tian, Jiwen Lu, Jianjiang Feng, and Jie Zhou.

Action recognition in RGB-D egocentric videos. ICIP, 2017. 20

[97] Yansong Tang, Zian Wang, Jiwen Lu, Jianjiang Feng, and Jie Zhou.
Multi-stream deep neural networks for RGB-D egocentric action
recognition. IEEE Transactions on Circuits and Systems for Video
Technology, 2019. 20

[98] The World Bank.

The world by income and regions,
https://datatopics.worldbank.org/world-development-

2022.
indicators/the-world-by-income-and-region.html. 18

[99] Sebastian Thrun. Is learning the n-th thing any easier than learning

the ﬁrst? NeurIPS, 1995. 12

[100] Cameron Trotter, Georgia Atkinson, Matt Sharpe, Kirsten Richard-
son, A. Stephen McGough, Nick Wright, Ben Burville, and Per
Berggren. NDD20: A large-scale few-shot dolphin dataset for
coarse and ﬁne-grained categorisation. arXiv:2005.13359, 2020.
9, 19, 20, 23, 24

[101] United States Environmental Protection Agency. Greenhouse Gas
Equivalencies Calculator. https://www.epa.gov/energy/greenhouse-
gas-equivalencies-calculator, 2022. 28

[102] Koen EA van de Sande, Jasper RR Uijlings, Theo Gevers, and
Arnold WM Smeulders. Segmentation as selective search for ob-
ject recognition. ICCV, 2011. 10

[103] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin.
Attention is all you need. NeurIPS, 2017. 5, 16

[104] Boying Wang, Libo Zhang, Longyin Wen, Xianglong Liu, and Yan-
jun Wu. Towards real-world prohibited item detection: A large-
scale x-ray benchmark. CVPR, 2021. 9, 19, 20

[105] Weiyao Wang, Matt Feiszli, Heng Wang, Jitendra Malik, and
Du Tran. Open-world instance segmentation: Exploiting pseudo
ground truth from learned pairwise afﬁnity. CVPR, 2022. 21
[106] Chao-Yuan Wu, Justin Johnson, Jitendra Malik, Christoph Feicht-
enhofer, and Georgia Gkioxari. Multiview compressive coding for
3D reconstruction. CVPR, 2023. 12

[107] Jianxiong Xiao, James Hays, Krista Ehinger, Aude Oliva, and An-
tonio Torralba. SUN database: Large-scale scene recognition from
abbey to zoo. CVPR, 2010. 20

[108] Saining Xie and Zhuowen Tu. Holistically-nested edge detection.

ICCV, 2015. 10

[109] Ning Xu, Brian Price, Scott Cohen, Jimei Yang, and Thomas S
Huang. Deep interactive object selection. CVPR, 2016. 4, 19
[110] Kaiyu Yang, Klint Qinami, Li Fei-Fei, Jia Deng, and Olga Rus-
sakovsky. Towards fairer datasets: Filtering and balancing the dis-
tribution of the people subtree in the imagenet hierarchy. Proceed-
ings of the 2020 conference on fairness, accountability, and trans-
parency, 2020. 8

[111] Lei Yang, Yan Zi Wei, Yisheng HE, Wei Sun, Zhenhang Huang,
Haibin Huang, and Haoqiang Fan.
iShape: A ﬁrst step towards
irregular shape instance segmentation. arXiv:2109.15068, 2021. 9,
20, 23, 24

[112] Senthil Yogamani, Ciar´an Hughes, Jonathan Horgan, Ganesh Sistu,
Padraig Varley, Derek O’Dea, Michal Uric´ar, Stefan Milz, Mar-
tin Simon, Karl Amende, et al. WoodScape: A multi-task, multi-
ICCV, 2019. 9,
camera ﬁsheye dataset for autonomous driving.
20

[113] Lingzhi Zhang, Shenghao Zhou, Simon Stent, and Jianbo Shi. Fine-
grained egocentric hand-object segmentation: Dataset, model, and
applications. ECCV, 2022. 9, 19, 20

[114] Wenwei Zhang, Jiangmiao Pang, Kai Chen, and Chen Change Loy.

K-Net: Towards uniﬁed image segmentation. NeurIPS, 2021. 4

[115] Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-
Wei Chang. Men also like shopping: Reducing gender bias ampli-
ﬁcation using corpus-level constraints. arXiv:1707.09457, 2017. 8
[116] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and An-
tonio Torralba. Places: A 10 million image database for scene
recognition. TPAMI, 2017. 20

[117] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler,
Adela Barriuso, and Antonio Torralba. Semantic understanding of
scenes through the ADE20K dataset. IJCV, 2019. 2, 7, 9, 20

15

Appendix

Table of contents:

• §A: Segment Anything Model and Task Details
• §B: Automatic Mask Generation Details
• §C: RAI Additional Details
• §D: Experiment Implementation Details
• §E: Human Study Experimental Design
• §F: Dataset, Annotation, and Model Cards
• §G: Annotation Guidelines

A. Segment Anything Model and Task Details

Image encoder. In general, the image encoder can be any
network that outputs a C×H×W image embedding. Mo-
tivated by scalability and access to strong pre-training, we
use an MAE [47] pre-trained Vision Transformer (ViT) [33]
with minimal adaptations to process high resolution inputs,
speciﬁcally a ViT-H/16 with 14×14 windowed attention
and four equally-spaced global attention blocks, follow-
ing [62]. The image encoder’s output is a 16× downscaled
embedding of the input image. Since our runtime goal is to
process each prompt in real-time, we can afford a high num-
ber of image encoder FLOPs because they are computed
only once per image, not per prompt.

Following standard practices (e.g., [40]), we use an in-
put resolution of 1024×1024 obtained by rescaling the im-
age and padding the shorter side. The image embedding
is therefore 64×64. To reduce the channel dimension, fol-
lowing [62], we use a 1×1 convolution to get to 256 chan-
nels, followed by a 3×3 convolution also with 256 channels.
Each convolution is followed by a layer normalization [4].

Prompt encoder. Sparse prompts are mapped to 256-
dimensional vectorial embeddings as follows. A point is
represented as the sum of a positional encoding [95] of the
point’s location and one of two learned embeddings that in-
dicate if the point is either in the foreground or background.
A box is represented by an embedding pair: (1) the posi-
tional encoding of its top-left corner summed with a learned
embedding representing “top-left corner” and (2) the same
structure but using a learned embedding indicating “bottom-
right corner”. Finally, to represent free-form text we use the
text encoder from CLIP [82] (any text encoder is possible in
general). We focus on geometric prompts for the remainder
of this section and discuss text prompts in depth in §D.5.

Dense prompts (i.e., masks) have a spatial correspon-
dence with the image. We input masks at a 4× lower res-
olution than the input image, then downscale an additional
4× using two 2×2, stride-2 convolutions with output chan-
nels 4 and 16, respectively. A ﬁnal 1×1 convolution maps
the channel dimension to 256. Each layer is separated by
GELU activations [50] and layer normalization. The mask

Figure 14: Details of the lightweight mask decoder. A
two-layer decoder updates both the image embedding and
prompt tokens via cross-attention. Then the image embed-
ding is upscaled, from which the updated output tokens are
used to dynamically predict masks. (Not illustrated for ﬁg-
ure clarity: At every attention layer, positional encodings
are added to the image embedding, and the entire original
prompt token (including position encoding) is re-added to
the token queries and keys.)

and image embedding are then added element-wise. If there
is no mask prompt, a learned embedding representing “no
mask” is added to each image embedding location.

Lightweight mask decoder. This module efﬁciently maps
the image embedding and a set of prompt embeddings to an
output mask. To combine these inputs, we take inspiration
from Transformer segmentation models [14, 20] and modify
a standard Transformer decoder [103]. Before applying our
decoder, we ﬁrst insert into the set of prompt embeddings
a learned output token embedding that will be used at the
decoder’s output, analogous to the [class] token in [33].
For simplicity, we refer to these embeddings (not including
the image embedding) collectively as “tokens”.

Our decoder design is shown in Fig. 14. Each decoder
layer performs 4 steps: (1) self-attention on the tokens, (2)
cross-attention from tokens (as queries) to the image em-
bedding, (3) a point-wise MLP updates each token, and (4)
cross-attention from the image embedding (as queries) to
tokens. This last step updates the image embedding with
prompt information. During cross-attention, the image em-
bedding is treated as a set of 642 256-dimensional vectors.
Each self/cross-attention and MLP has a residual connec-
tion [49], layer normalization, and a dropout [93] of 0.1 at
training. The next decoder layer takes the updated tokens
and the updated image embedding from the previous layer.
We use a two-layer decoder.

To ensure the decoder has access to critical geometric in-
formation the positional encodings are added to the image
embedding whenever they participate in an attention layer.
Additionally, the entire original prompt tokens (including
their positional encodings) are re-added to the updated to-
kens whenever they participate in an attention layer. This
allows for a strong dependence on both the prompt token’s
geometric location and type.

After running the decoder, we upsample the updated im-
age embedding by 4× with two transposed convolutional

16

imageembedding(256x64x64)x2tokento imageattn.2xconv.trans.IoUscoresmlpmasksdot productper maskprompt tokens(Ntokensx256)output tokens+outputtokenper maskIoUoutputtokenmlpmask decoderself attn.token to image attn.mlpimage to token attn.layers (now it’s downscaled 4× relative to the input image).
Then, the tokens attend once more to the image embedding
and we pass the updated output token embedding to a small
3-layer MLP that outputs a vector matching the channel di-
mension of the upscaled image embedding. Finally, we pre-
dict a mask with a spatially point-wise product between the
upscaled image embedding and the MLP’s output.

The transformer uses an embedding dimension of 256.
The transformer MLP blocks have a large internal dimen-
sion of 2048, but the MLP is applied only to the prompt to-
kens for which there are relatively few (rarely greater than
20). However, in cross-attention layers where we have a
64×64 image embedding, we reduce the channel dimension
of the queries, keys, and values by 2× to 128 for computa-
tional efﬁciency. All attention layers use 8 heads.

The transposed convolutions used to upscale the output
image embedding are 2×2, stride 2 with output channel di-
mensions of 64 and 32 and have GELU activations. They
are separated by layer normalization.

Making the model ambiguity-aware. As described, a sin-
gle input prompt may be ambiguous in the sense that it cor-
responds to multiple valid masks, and the model will learn
to average over these masks. We eliminate this problem
with a simple modiﬁcation: instead of predicting a single
mask, we use a small number of output tokens and predict
multiple masks simultaneously. By default we predict three
masks, since we observe that three layers (whole, part, and
subpart) are often enough to describe nested masks. During
training, we compute the loss (described shortly) between
the ground truth and each of the predicted masks, but only
backpropagate from the lowest loss. This is a common tech-
nique used for models with multiple outputs [15, 45, 64].
For use in applications, we’d like to rank predicted masks,
so we add a small head (operating on an additional output
token) that estimates the IoU between each predicted mask
and the object it covers.

Ambiguity is much rarer with multiple prompts and the
three output masks will usually become similar. To mini-
mize computation of degenerate losses at training and en-
sure the single unambiguous mask receives a regular gradi-
ent signal, we only predict a single mask when more than
one prompt is given. This is accomplished by adding a
fourth output token for an additional mask prediction. This
fourth mask is never returned for a single prompt and is the
only mask returned for multiple prompts.

Losses. We supervise mask prediction with a linear combi-
nation of focal loss [65] and dice loss [73] in a 20:1 ratio of
focal loss to dice loss, following [20, 14]. Unlike [20, 14],
we observe that auxiliary deep supervision after each de-
coder layer is unhelpful. The IoU prediction head is trained
with mean-square-error loss between the IoU prediction and
the predicted mask’s IoU with the ground truth mask. It is
added to the mask loss with a constant scaling factor of 1.0.

Training algorithm. Following recent approaches [92, 37],
we simulate an interactive segmentation setup during train-
ing. First, with equal probability either a foreground point
or bounding box is selected randomly for the target mask.
Points are sampled uniformly from the ground truth mask.
Boxes are taken as the ground truth mask’s bounding box,
with random noise added in each coordinate with standard
deviation equal to 10% of the box sidelength, to a maxi-
mum of 20 pixels. This noise proﬁle is a reasonable com-
promise between applications like instance segmentation,
which produce a tight box around the target object, and in-
teractive segmentation, where a user may draw a loose box.
After making a prediction from this ﬁrst prompt, subse-
quent points are selected uniformly from the error region
between the previous mask prediction and the ground truth
mask. Each new point is foreground or background if the er-
ror region is a false negative or false positive, respectively.
We also supply the mask prediction from the previous it-
eration as an additional prompt to our model. To provide
the next iteration with maximal information, we supply the
unthresholded mask logits instead of the binarized mask.
When multiple masks are returned, the mask passed to the
next iteration and used to sample the next point is the one
with the highest predicted IoU.

We ﬁnd diminishing returns after 8 iteratively sampled
points (we have tested up to 16). Additionally, to encour-
age the model to beneﬁt from the supplied mask, we also
use two more iterations where no additional points are sam-
pled. One of these iterations is randomly inserted among the
8 iteratively sampled points, and the other is always at the
end. This gives 11 total iterations: one sampled initial in-
put prompt, 8 iteratively sampled points, and two iterations
where no new external information is supplied to the model
so it can learn to reﬁne its own mask predictions. We note
that using a relatively large number of iterations is possible
because our lightweight mask decoder requires less than 1%
of the image encoder’s compute and, therefore, each itera-
tion adds only a small overhead. This is unlike previous
interactive methods that perform only one or a few interac-
tive steps per optimizer update [70, 9, 37, 92].

Training recipe. We use the AdamW [68] optimizer (β1 =
0.9, β2 = 0.999) and a linear learning rate warmup [42] for
250 iterations and a step-wise learning rate decay schedule.
The initial learning rate (lr), after warmup, is 8e−4. We
train for 90k iterations (∼2 SA-1B epochs) and decrease the
lr by a factor of 10 at 60k iterations and again at 86666 it-
erations. The batch size is 256 images. To regularize SAM,
we set weight decay (wd) to 0.1 and apply drop path [53]
(dp) with a rate of 0.4. We use a layer-wise learning rate
decay [5] (ld) of 0.8. No data augmentation is applied. We
initialize SAM from an MAE [47] pre-trained ViT-H. We
distribute training across 256 GPUs, due to the large image
encoder and 1024×1024 input size. To limit GPU mem-

17

ory usage, we train with up to 64 randomly sampled masks
per GPU. Additionally, we ﬁnd that lightly ﬁltering SA-1B
masks to discard any that cover more than 90% of the image
qualitatively improves results.

For ablations and others variations on training (e.g., text-
to-mask §D.5), we deviate from the default recipe above as
follows. When training with data from the ﬁrst and sec-
ond data engine stages only, we augment the input with
large-scale jitter [40] with a scale range of [0.1, 2.0]. In-
tuitively, data augmentation may be helpful when training
data is more limited. To train ViT-B and ViT-L, we use
180k iterations with batch size 128 distributed across 128
GPUs. We set lr = 8e−4/4e−4, ld = 0.6/0.8, wd = 0.1, and
dp = 0.6/0.4 for ViT-B/L, respectively.

B. Automatic Mask Generation Details

Here we discuss details of the data engine’s fully auto-

matic stage that was used to generate the released SA-1B.

Cropping. Masks were generated from a regular grid of
32×32 points on the full image and 20 additional zoomed-
in image crops arising from 2×2 and 4×4 partially over-
lapping windows using 16×16 and 8×8 regular point grids,
respectively. The original high-resolution images were used
for cropping (this was the only time we used them). We re-
moved masks that touch the inner boundaries of the crops.
We applied standard greedy box-based NMS (boxes were
used for efﬁciency) in two phases: ﬁrst within each crop and
second across crops. When applying NMS within a crop,
we used the model’s predicted IoU to rank masks. When
applying NMS across crops, we ranked masks from most
zoomed-in (i.e., from a 4×4 crop) to least zoomed-in (i.e.,
the original image), based on their source crop.
In both
cases, we used an NMS threshold of 0.7.

Filtering. We used three ﬁlters to increase mask qual-
ity. First, to keep only conﬁdent masks we ﬁltered by the
model’s predicted IoU score at a threshold of 88.0. Second,
to keep only stable masks we compared two binary masks
resulting from the same underlying soft mask by threshold-
ing it at different values. We kept the prediction (i.e., the
binary mask resulting from thresholding logits at 0) only if
the IoU between its pair of -1 and +1 thresholded masks was
equal to or greater than 95.0. Third, we noticed that occa-
sionally an automatic mask would cover the entire image.
These masks were generally uninteresting, and we ﬁltered
them by removing masks that covered 95% or more of an
image. All ﬁltering thresholds were selected to achieve both
a large number of masks and high mask quality as judged by
professional annotators using the method described in §5.

than 100 pixels (including removing entire masks if the
largest component is below this threshold). Second, another
estimated 4% of masks include small, spurious holes. To
address these, we ﬁlled holes with area less than 100 pixels.
Holes were identiﬁed as components of inverted masks.

Automatic mask generation model. We trained a special
version of SAM for fully automatic mask generation that
sacriﬁces some inference speed for improved mask gener-
ation properties. We note the differences between our de-
fault SAM and the one used for data generation here:
it
was trained on manual and semi-automatic data only, it was
trained for longer (177656 iterations instead of 90k) with
large-scale jitter data augmentation [40], simulated interac-
tive training used only point and mask prompts (no boxes)
and sampled only 4 points per mask during training (reduc-
ing from our default of 9 to 4 sped up training iterations
and had no impact on 1-point performance, though it would
harm mIoU if evaluating with more points), and ﬁnally the
mask decoder used 3 layers instead of 2.

SA-1B examples. We show SA-1B samples in Fig. 2. For
more examples, please see our dataset explorer.

C. RAI Additional Details

Inferring geographic information for SA-1B. While the
images in SA-1B are not geo-tagged, each image has a cap-
tion describing its contents and where it was taken. We infer
approximate image geo-locations from these captions using
an Elmo-based named entity recognition model [78]. Each
extracted location entity is mapped to every matching coun-
try, province, and city. Captions are mapped to a single
country by ﬁrst considering the matching countries, then
provinces, and ﬁnally cities. We note that there are ambigu-
ities and potential for biases with this method (e.g., “Geor-
gia” may refer to the country or the US state). As such, we
use the extracted locations to analyze the dataset as a whole,
but do not release the inferred locations. The captions will
not be released publicly as required by the image provider.

Inferring geographic information for COCO and Open
Images. The COCO [66] and Open Images [60] datasets
do not provide geo-locations. Following [29], we retrieve
geographic metadata using the Flickr API. We retrieved
locations for 24% of the COCO training set (19,562 im-
ages) and for Open Images we retrieved 18% of the train-
ing set (493,517 images, after only considering images with
masks). We note that the geographic information is approx-
imate, and the sample of images with this information may
not fully match the full dataset distribution.

Postprocessing. We observed two error types that are eas-
ily mitigated with postprocessing. First, an estimated 4%
of masks include small, spurious components. To address
these, we removed connected components with area less

Inferring income information. We use each image’s in-
ferred country to look up its income level using the levels
deﬁned by The World Bank [98]. We collapse the upper-
middle and lower-middle levels into a single middle level.

18

mIoU at

mIoU at

1 point

3 points

1 point

3 points

perceived gender presentation
feminine
76.3 ±1.1 90.7 ±0.5
masculine 81.0 ±1.2 92.3 ±0.4

perceived age group
older
middle
young

81.9 ±3.8
78.2 ±0.8
77.3 ±2.7

92.8 ±1.6
91.3 ±0.3
91.5 ±0.9

Table 6: SAM’s performance segmenting clothing across
perceived gender presentation and age group. The intervals
for perceived gender are disjoint, with mIoU for masculine
being higher. Conﬁdence intervals for age group overlap.

Fairness in segmenting people. To investigate SAM’s fair-
ness at segmenting people we use the More Inclusive Anno-
tations for People (MIAP) [87] test set annotations for Open
Images [60], which allows us to compare SAM’s perfor-
mance across perceived gender presentation and perceived
age group. MIAP provides box annotations, while we need
ground truth masks for this analysis. To get ground truth
masks, we select each person-category mask from Open
Images if its corresponding bounding box is within a 1%
margin (based on relative box side lengths) of an annotated
bounding box in MIAP, resulting in 3.9k masks.

Fairness in segmenting clothing. We extend our analysis
from §6 to clothing segmentation. We look at SAM’s per-
formance on clothing relative to the attributes of those wear-
ing the clothes. We use all 6.5k ground truth masks from
Open Images that have a category under the clothing super-
class and reside within a person box from MIAP. In Table 6
we compare performance across perceived gender presenta-
tion and age group. We ﬁnd that SAM is better at segment-
ing clothing on those who present predominantly mascu-
line, with disjoint 95% conﬁdence intervals. The gap closes
when moving from 1 to 3 point evaluation. Differences for
perceived age group are not signiﬁcant. Our results indicate
there is a bias when segmenting clothing across perceived
gender presentation with a one point prompt, and we en-
courage users of SAM to be mindful of this limitation.

D. Experiment Implementation Details

D.1. Zero-Shot Single Point Valid Mask Evaluation

Datasets. We built a new segmentation benchmark to eval-
uate the zero-shot transfer capabilities of our model using a
suite of 23 diverse segmentation datasets from prior work.
A description of each dataset is given in Table 7. For exam-
ples, see main text Fig. 8. This suite covers a range of do-
mains including egocentric [34, 28, 113], microscopy [12],
X-ray [104], underwater [52, 100], aerial [17], simula-
tion [86], driving [25], and painting [24] images. For ef-
ﬁcient evaluation we subsampled datasets with more than
15k masks. Speciﬁcally, we randomly picked images so
that the total number of masks in the sampled images was
∼10k. We blurred faces of people in all the datasets.

Point sampling. Our default point sampling follows stan-
dard practice in interactive segmentation [109, 64, 92]. The
ﬁrst point is chosen deterministically as the point farthest
from the object boundary. Each subsequent point is the
farthest from the boundary of the error region between
ground truth and the previous prediction. Some experiments
(where speciﬁed) use a more challenging sampling strategy
in which the ﬁrst point is a random point, rather than a deter-
ministically selected “center” point. Each subsequent point
is selected as described above. This setting better reﬂects
use cases in which the ﬁrst point is not reliably near the
center of the mask, such as prompting from eye gaze.

Evaluation. We measure IoU between a prediction after
N point prompts and a ground truth mask, where N =
{1, 2, 3, 5, 9} and points are sampled iteratively with either
of the strategies described above. The per-dataset mIoU is
the per-mask IoU averaged across all objects in the dataset.
Finally, we report the top-line metric by averaging the per-
dataset mIoUs across all 23 datasets. Our evaluation differs
from the standard interactive segmentation evaluation pro-
tocol which measures the average number of points needed
to achieve X% IoU, with up to 20 points. We focus on pre-
dictions after just one, or possibly a few points, since many
of our use cases involve a single or very few prompts. Given
our application focus, which requires real-time prompt pro-
cessing, we expect the best interactive segmentation models
to outperform SAM when using a large number of points.

Baselines. We use three recent strong interactive base-
lines: RITM [92], FocalClick [18], and SimpleClick [67].
For each, we use the largest models trained on the broad-
est datasets publicly released by the authors. For RITM,
we use HRNet32 IT-M trained on the combination of
COCO [66] and LVIS [44] introduced by the authors.
For FocalClick, we use SegFormerB3-S2 trained on a
“combined dataset” that includes 8 different segmentation
datasets [18]. For SimpleClick, we use ViT-H448 trained
on a combination of COCO and LVIS. We follow the sug-
gested default strategies for data pre-processing (i.e., data
augmentations or image resizing) and do not change or
adapt any parameters for our evaluation.
In our experi-
ments, we observe that RITM outperforms other baselines
on our 23 dataset suite with 1 point evaluation. Therefore,
we use RITM as the default baseline. When evaluating with
more points we report results for all baselines.

Single point ambiguity and oracle evaluation. In addition
to IoU after N points prompts, we report SAM’s “oracle”
performance at 1 point by evaluating the predicted mask that
best matches ground truth from amongst SAM’s three pre-
dictions (rather than using the one that SAM itself ranks
ﬁrst, as we do by default). This protocol addresses possible
single point prompt ambiguity by relaxing the requirement
to guess the one right mask among several valid objects.

19

dataset

Plant Phenotyping Datasets
Leaf Segmentation [74]

BBBC038v1 from Broad
Bioimage Benchmark
Collection [12]

abbreviation
& link

image
type

description

mask
type

source split

# images
sampled

# masks
sampled

PPDLS Plants

Leaf segmentation for images of tobacco and ara plants.

Instance N/A

182

2347

BBBC038v1 Microscopy

Biological images of cells in a variety of settings testing
robustness in nuclei segmentation.

Instance

Train

227

10506

Dataset fOr bOuldeRs
Segmentation [80]

DOORS Boulders

Segmentation masks of single boulders positioned on the
surface of a spherical mesh.

Instance DS1

10000

10000

TimberSeg 1.0 [38]

TimberSeg Logs

Segmentation masks of individual logs in piles of timber in
various environments and conditions. Images are taken from
an operator’s point-of-view.

Instance N/A

220

2487

Northumberland Dolphin
Dataset 2020 [100]

NDD20 Underwater

Segmentation masks of two different dolphin species in
images taken above and under water.

Instance N/A

4402

6100

Large Vocabulary Instance
Segmentation [44]

STREETS [91]

LVIS Scenes

STREETS

Trafﬁc
camera

Additional annotations for the COCO [66] dataset to enable
the study of long-tailed object detection and segmentation.

Instance Validation (v0.5)

Segmentation masks of cars in trafﬁc camera footage.

Instance N/A

945

819

9642

9854

ZeroWaste-f [6]

ZeroWaste-f Recycling

Segmentation masks in cluttered scenes of deformed
recycling waste.

Instance

Train

2947

6155

iShape [111]

iShape

Irregular
shapes

Segmentation masks of irregular shapes like antennas, logs,
fences, and hangers.

Instance Validation

754

9742

ADE20K [117]

ADE20K Scenes

Object and part segmentation masks for images from
SUN [107] and Places [116] datasets.

Instance Validation

302

10128

Occluded Video Instance
Segmentation [81]

OVIS Occlusions

Instance segmentation masks in videos, focusing on objects
that are occluded.

Instance

Train

2044

10011

Hypersim [86]

Hypersim Simulation

Photorealistic synthetic dataset of indoor scenes with instance
masks.

Instance

Evermotion archinteriors
volumes 1-55 excluding
20,25,40,49

338

9445

Night and Day Instance
Segmented Park [22, 23]

NDISPark Parking lots

Images of parking lots from video footage taken at day and
night during different weather conditions and camera angles
for vehicle segmentation.

Instance

Train

111

2577

EPIC-KITCHENS
VISOR [28, 27]

VISOR Egocentric

Segmentation masks for hands and active objects in
ego-centric video from the cooking dataset
EPIC-KITCHENS [27].

Instance Validation

1864

10141

Plittersdorf dataset [46]

Plittersdorf

Stereo
images

Segmentation masks of wildlife in images taken with the
SOCRATES stereo camera trap.

Instance

Train, validation, test

187

546

Egocentric Hand-Object
Segmentation [113]

EgoHOS Egocentric

Fine-grained egocentric hand-object segmentation dataset.
Dataset contains mask annotations for existing datasets.

Instance

Train (including only
Ego4D [43] and
THU-READ [97, 96])

2940

9961

InstanceBuilding 2D [17]

IBD Drones

High-resolution drone UAV images annotated with roof
instance segmentation masks.

Instance

Train (2D annotations)

467

11953

WoodScape [112]

WoodScape

Fisheye
driving

Fisheye driving dataset with segmentation masks. Images are
taken from four surround-view cameras.

Instance

Set 1

Cityscapes [25]

Cityscapes Driving

Stereo video of street scenes with segmentation masks.

Panoptic Validation

107

293

10266

9973

PIDray [104]

PIDRay X-ray

Segmentation masks of prohibited items in X-ray images of
baggage.

Instance

Test (hard)

3733

8892

Diverse Realism in Art
Movements [24]

DRAM Paintings

Domain adaptation dataset for semantic segmentation of art
paintings.

Semantic Test

718

1179

TrashCan [52]

TrashCan Underwater

Georgia Tech Egocentric
Activity Datasets [34, 63]

GTEA Egocentric

Segmentation masks of trash in images taken by underwater
ROVs. Images are sourced from the J-EDI [69] dataset.

Videos are composed of four different subjects performing
seven types of daily activities with segmentation masks of
hands.

Instance

Train (instance task)

5936

9540

Instance

Train (segmenting hands
task)

652

1208

Table 7: Segmentation datasets used to evaluate zero-shot segmentation with point prompts. The 23 datasets cover a broad
range of domains; see column “image type”. To make our evaluation efﬁcient, we subsample datasets that have more than
15k masks. Speciﬁcally, we randomly sampled images so that the total number of masks in the images is ∼10k.

20

image

ground truth

SAM

image

ground truth

SAM

Figure 15: Additional visualizations of zero-shot edge predictions on BSDS500. Recall that SAM was not trained to predict
edge maps and did not have access to BSDS images and annotations during training.

D.2. Zero-Shot Edge Detection

Dataset and metrics. We perform zero-shot edge detection
experiments on BSDS500 [72, 3]. The ground truth for each
image comes from the manual annotations of ﬁve different
subjects. We report results on the 200 image test subset
using the four standard metrics for edge detection [3, 32]:
optimal dataset scale (ODS), optimal image scale (OIS), av-
erage precision (AP), and recall at 50% precision (R50).

Method. For zero-shot transfer, we use a simpliﬁed ver-
sion of our automatic mask generation pipeline. We prompt
SAM with a 16×16 regular grid of foreground points,
which yields 768 predicted masks (three per point). We do
not ﬁlter by predicted IoU or stability. Redundant masks
are removed by NMS. Then we apply a Sobel ﬁlter to the
remaining masks’ unthresholded probability maps and set
values to zero if they do not intersect with the outer bound-
ary pixels of a mask. Finally, we take a pixel-wise max over
all the predictions, linearly normalize the result to [0,1], and
apply edge NMS [13] to thin the edges.

Visualizations. In Fig. 15, we show additional examples
of zero-shot edge predictions from SAM. These qualitative
examples further illustrate how SAM tends to output sensi-
ble edge maps, despite not being trained for edge detection.
We see that the edges can align well with the human anno-
tations. Although, as previously mentioned, since SAM is
not trained for edge detection it does not learn the biases of
the BSDS500 dataset and often outputs more edges than are
present in the ground truth annotations.

D.3. Zero-Shot Object Proposals

Dataset and metrics. We report the standard average recall
(AR) metric for masks at 1000 proposals on the LVIS v1
validation set [44]. Since LVIS has high-quality masks for
1203 object classes, it provides a challenging test for ob-
ject proposal generation. We focus on AR@1000 due to the
open-world nature of our model, which will likely produce
many valid masks outside even the 1203 classes in LVIS. To
measure performance on frequent, common, and rare cate-

gories, we use AR@1000 but measured against a ground
truth set containing just the corresponding LVIS categories.

Baseline. We use cascade ViTDet-H as a baseline, the
strongest model from [62] by AP on LVIS. As noted in the
main text, an object detector trained in-domain can “game”
AR [16] and is expected to be a stronger baseline than other
models that focus on open-world proposals or segmenta-
tion [58, 105]. To produce 1000 proposals, we disable score
thresholding in the three cascade stages and as raise the
maximum number of predictions per stage to 1000.

Method. We use a modiﬁed version of SAM’s automatic
mask generation pipeline for zero-shot transfer. First, to
make inference time comparable to that of ViTDet we do
not process image crops. Second, we remove ﬁltering by
predicted IoU and stability. This leaves two tunable param-
eters to get ∼1000 masks per image: the input point grid and
the NMS threshold duplicate mask suppression. We choose
a 64×64 point grid and an NMS threshold of 0.9, which
produces ∼900 masks per image on average. At evaluation,
if greater than 1000 masks have been proposed in an im-
age, they are ranked by the average of their conﬁdence and
stability scores, then truncated to the top 1000 proposals.

We hypothesize that SAM’s ability to output multiple
masks is especially valuable for this task, since recall should
beneﬁt from proposals generated at multiple scales from
a single input point. To test this, we compare to an ab-
lated version SAM that only outputs a single mask instead
of three (SAM - single-output). Since this model produces
fewer masks, we further increase the number of points sam-
pled and NMS threshold to 128×128 and 0.95, respectively,
obtaining ∼950 masks per image on average. Additionally,
single-output SAM does not produce the IoU score used
to rank masks for NMS in the automatic mask generation
pipeline, so instead masks are ranked randomly. Testing
suggests this has similar performance to more sophisticated
methods of ranking masks, such as using the max logit value
of the mask as a proxy for model conﬁdence.

21

ground truth

ViTDet

SAM

ground truth

ViTDet

SAM

Figure 16: Zero-shot instance segmentation on LVIS v1. SAM produces higher quality masks than ViTDet. As a zero-shot
model, SAM does not have the opportunity to learn speciﬁc training data biases; see top-right as an example where SAM
makes a modal prediction, whereas the ground truth in LVIS is amodal given that mask annotations in LVIS have no holes.

D.4. Zero-Shot Instance Segmentation

Method. For zero-shot instance segmentation, we prompt
SAM with the boxes output by a fully-supervised ViTDet-H
on COCO and LVIS v1 validation splits. We apply an ad-
ditional mask reﬁnement iteration by feeding the most con-
ﬁdent predicted mask, together with the box prompt, back
to the mask decoder to produce the ﬁnal prediction. We
show zero-shot instance segmentations predicted on LVIS
in Fig. 16. Compared to ViTDet, SAM tends to produce
higher quality masks with cleaner boundaries. We conﬁrm
this observation with human studies in §7.4. Note that as a
zero-shot model, SAM is not able to learn annotation biases
in a dataset. For instance, we see that SAM makes a valid
modal prediction for the plate, whereas LVIS masks cannot
contain holes by design so the plate is annotated amodally.

D.5. Zero-Shot Text-to-Mask

Model and training. We use the largest publicly available
CLIP model [82] (ViT-L/14@336px) to compute text
and image embeddings, which we (cid:96)2 normalize prior to use.
To train SAM, we use masks from the ﬁrst two stages of our
data engine. Moreover, we discard all masks with an area
smaller than 1002 pixels. We train this model with large-
scale jitter [40] for 120k iterations with batch size 128. All
other training parameters follow our default settings.

Generating training prompts. To extract an input prompt
we ﬁrst expand the bounding box around each mask by a
random factor from 1× to 2×, square-crop the expanded
box to maintain its aspect ratio, and resize it to 336×336
pixels. Before feeding the crop to the CLIP image encoder,
with 50% probability we zero-out pixels outside the mask.
To ensure the embedding focuses on the object, we use
masked attention in the last layer to restrict attention from
the output token to the image positions inside the mask. Fi-
nally, our prompt is the output token embedding. For train-
ing we supply the CLIP-based prompt ﬁrst, followed by ad-
ditional iterative point prompts to reﬁne the prediction.

Figure 17: Visualization of thresholding the similarities of
mask embeddings from SAM’s latent space. A query is in-
dicated by the magenta box; top row shows matches at a low
threshold, bottom row at a high threshold. The most similar
mask embeddings in the same image can often be seman-
tically similar to the query mask embedding, even though
SAM is not trained with explicit semantic supervision.

Inference. During inference we use the CLIP text encoder
without any modiﬁcations to create a prompt for SAM. We
rely on the fact that text and image embeddings are aligned
by CLIP, which allows us to train without any explicit text
supervision while using text-based prompts for inference.

D.6. Probing the Latent Space of SAM

Finally, we perform an initial investigation to qualita-
tively probe the latent space learned by SAM. In particu-
lar, we are interested in whether SAM is able to capture any
semantics in its representation even though is not trained
with explicit semantic supervision. To do so, we compute
mask embeddings by extracting an image embedding from
SAM from an image crop around a mask and its horizon-
tally ﬂipped version, multiplying the image embedding by
the binary mask, and averaging over spatial locations. In
Fig. 17, we show 3 examples of a query mask and similar
masks (in the latent space) in the same image. We observe

22

that the nearest neighbors for each query show some, albeit
imperfect, shape and semantic similarity. Although these
results are preliminary, they indicate that the representations
from SAM may be useful for a variety of purposes, such as
further data labeling, understanding the contents of datasets,
or as features for downstream tasks.

E. Human Study Experimental Design

Here we describe details of the human study used to eval-
uate mask quality in §7.1 and §7.4. The purpose of the
human study is to address two limitations of using IoU to
ground truth as a measure of predicted mask quality. The
ﬁrst limitation is that, for ambiguous inputs such as a single
point, the model may be strongly penalized for returning a
valid mask of a different object than the ground truth. The
second limitation is that ground truth masks may include
various biases, such as systematic errors in the edge qual-
ity or decisions to modally or amodally segment occluding
objects. A model trained in-domain can learn these biases
and obtain a higher IoU without necessarily producing bet-
ter masks. Human review can obtain a measure of mask
quality independent of an underlying ground truth mask in
order to alleviate these issues.

Models. For single-point evaluation, we use RITM [92],
single-output SAM, and SAM to test two hypotheses. First,
we hypothesize that SAM produces visually higher quality
masks than baseline interactive segmentation models when
given a single point, even when metrics such as IoU with
ground truth do not reveal this. Second, we hypothesize
that SAM’s ability to disambiguate masks improves mask
quality for single point inputs, since single output SAM may
return masks that average over ambiguous masks.

For instance segmentation experiments, we evaluate cas-
cade ViTDet-H [62] and SAM in order to test the hypothesis
that SAM produces visually higher quality masks, even if it
obtains a lower AP due to the inability to learn speciﬁc an-
notation biases of the validation dataset.

Datasets. For single-point experiments, we select 7 datasets
from our set of 23 datasets, since the full suite is too large
for human review. We choose LVIS v0.5 [17], VISOR [28,
27], DRAM [24], IBD [17], NDD20 [100], OVIS [81], and
iShape [111], which provide a diverse collection of images,
including scene-level, ego-centric, drawn, overhead, under-
water, and synthetic imagery. Additionally, this set includes
datasets both where SAM outperforms RITM with IoU met-
rics and vice-versa. For instance segmentation experiments,
we use the LVIS v1 validation set, allowing for direct com-
parison to ViTDet, which was trained on LVIS.

Methodology. We presented masks generated by the mod-
els to professional annotators and asked them to rate each
mask using provided guidelines (see §G for the complete
guidelines). Annotators were sourced from the same com-

pany that collected manually annotated masks for the data
engine. An annotator was provided access to an image, the
predicted mask of a single model, and the input to the model
(either a single point or single box) and asked to judge the
mask on three criterion: Does the mask correspond to a
valid object? Does the mask have a clean boundary? and
Does the mask correspond to the input? They then submit-
ted a rating from 1-10 indicating the overall mask quality.

A score of 1 indicates a mask that corresponds to no ob-
ject at all; a low score (2-4) indicates that the mask has huge
errors, such including huge regions of other objects or hav-
ing large areas of nonsensical boundaries; a middle score
(5-6) indicates masks that are mostly sensible but still have
signiﬁcant semantic or boundary errors; a high score (7-
9) indicates masks with only minor boundary errors; and a
score of 10 is for masks with no visible errors. Annotators
were provided with ﬁve different views, each designed to
help identify different error types.

For single point experiments, 1000 masks per dataset
were selected randomly from the same subsets used for
benchmarking zero-shot interactive segmentation (see §D.1
for details on these subsets). The model input was the cen-
termost point, calculated as the largest value of the distance
transform from the edge of the mask. For instance seg-
mentation experiments, 1000 masks were selected from the
LVIS v1 validation set, and the model input was the LVIS
ground truth box.
In all experiments, masks with a size
smaller than 242 pixels were excluded from sampling, to
prevent showing raters a mask that was too small to judge
accurately. For both memory and display reasons, large im-
ages were rescaled to have a max side-length of 2000 before
predicting a mask. In all experiments, the same inputs were
fed to each model to produce a predicted mask.

For comparison,

the ground truth masks from each
dataset were also submitted for rating. For single-point
experiments, this gave 4000 total rating jobs per dataset
(1000 masks each for RITM, SAM single-output, SAM,
and ground truth); for instance segmentation experiments,
it gave 3000 total jobs (ViTDet, SAM, and ground truth).

For each dataset, these jobs were inserted with random
ordering into a queue from which 30 annotators drew jobs.
In initial testing of the review study, we provided each job to
ﬁve different annotators and found reasonable consistency
in scores: the average standard deviation in score over the
ﬁve annotators was 0.83. Additionally, the annotation com-
pany deployed quality assurance testers who spot checked
a fraction of results for extreme departures from the guide-
lines. Thus for our experiments each job (i.e., rating one
mask in one image) was completed by only a single anno-
tator. Average time spent per annotator per job was 90 sec-
onds, longer than our initial target of 30 seconds, but still
sufﬁciently fast to collect a large number of ratings on each
of the 7 selected datasets.

23

(a) LVIS v0.5 [17]

(b) VISOR [28, 27]

(c) DRAM [24]

(d) IBD [17]

(e) NDD20 [100]

(f) OVIS [81]

(g) iShape [111]

Figure 18: Mask quality rating distributions by dataset from our human evaluation study.

SAM > baseline

CI99(∆µ)

p-value

dataset
point input (RITM [92] baseline):
LVIS v0.5 [44]
VISOR [28, 27]
DRAM [24]
IBD [17]
NDD20 [100]
OVIS [81]
iShape [111]
box input (ViTDet-H [62] baseline):
LVIS v1 [44]

4e-69
7e-98
1e-76
2e-57
2e-86
2e-64
2e-88

2e-05

(1.40, 1.84)
(1.81, 2.24)
(1.54, 2.00)
(1.03, 1.39)
(1.88, 2.37)
(1.38, 1.84)
(1.97, 2.47)

(0.11, 0.42)

SAM > SAM single out.
CI99(∆µ)

p-value

2e-11
7e-26
2e-24
1e-15
5e-08
3e-10
7e-23

(0.29, 0.64)
(0.58, 0.94)
(0.62, 1.03)
(0.32, 0.62)
(0.19, 0.55)
(0.27, 0.63)
(0.65, 1.10)

N/A

N/A

Table 8: Statistical tests showing signiﬁcance that SAM has
higher mask quality ratings than baseline and single-output
SAM. P-values are calculated by paired t-test, while conﬁ-
dence intervals for the difference in mean scores are calcu-
lated by paired bootstrap on 10k samples. All p-values are
signiﬁcant, and all conﬁdence intervals exclude zero.

Results. Fig. 18 shows histograms over ratings for each
dataset in the single-point experiments. We run statistical

24

tests for two hypotheses: (1) that SAM gets higher scores
than the baseline model (RITM or ViTDet) and (2) that
SAM gets higher scores than single-output SAM. P-values
are calculated via a paired t-test on the means of the model
scores, which we supplement with a paired bootstrap test on
10k samples to ﬁnd the 99% conﬁdence interval for the dif-
ference of means. Table 8 shows p-values and conﬁdence
intervals for these tests. All statistical tests are strongly sig-
niﬁcant, and all conﬁdence intervals exclude zero.

For instance segmentation, Fig. 11 of the main text
shows the histogram for ratings. To compare to COCO
ground truth, we additionally include 794 ratings of COCO
ground truth masks that were collected during our testing of
the human review process. These masks were presented to
raters using an identical setup as the LVIS results. For fair
comparison, results for LVIS in Fig. 11 were subsampled
to the same 794 inputs for each model and ground truth.
For Table 8, the full 1000 ratings are used to run statistical
tests, which show that SAM’s mask quality improvement
over ViTDet is statistically signiﬁcant.

12345678910Mask quality rating02040Percent of ratings6.5 ± 0.15, RITM7.7 ± 0.12, SAM - single output8.1 ± 0.10, SAM8.5 ± 0.09, GT12345678910Mask quality rating02040Percent of ratings6.3 ± 0.16, RITM7.5 ± 0.13, SAM - single output8.3 ± 0.09, SAM8.5 ± 0.13, GT12345678910Mask quality rating02040Percent of ratings5.9 ± 0.14, RITM6.8 ± 0.15, SAM - single output7.7 ± 0.13, SAM8.0 ± 0.15, GT12345678910Mask quality rating02040Percent of ratings7.1 ± 0.12, RITM7.9 ± 0.11, SAM - single output8.3 ± 0.08, SAM8.4 ± 0.09, GT12345678910Mask quality rating02040Percent of ratings6.4 ± 0.17, RITM8.2 ± 0.11, SAM - single output8.6 ± 0.10, SAM8.9 ± 0.06, GT12345678910Mask quality rating02040Percent of ratings6.1 ± 0.15, RITM7.7 ± 0.12, SAM - single output7.2 ± 0.13, SAM8.8 ± 0.09, GT12345678910Mask quality rating02040Percent of ratings4.9 ± 0.16, RITM6.2 ± 0.17, SAM - single output7.1 ± 0.15, SAM9.3 ± 0.06, GTF. Dataset, Annotation, and Model Cards

In §F.1 we provide a Dataset Card for SA-1B, follow-
ing [39], in a list of questions and answers. Next, we pro-
vide a Data Annotation Card in §F.2 for the ﬁrst two stages
of our data engine described in §4, following CrowdWork-
Sheets [30], again as a list of questions and answers. We
provide a Model Card following [75] in Table 9.

F.1. Dataset Card for SA-1B

Motivation

1. For what purpose was the dataset created? Was there a speciﬁc task in
mind? Was there a speciﬁc gap that needed to be ﬁlled? Please provide a
description. The contributions of our dataset to the vision community are
fourfold: (1) We release a dataset of 11M images and 1.1B masks, by far the
largest segmentation dataset to date. (2) The dataset we release is privacy
protecting: we have blurred faces and license plates in all images. (3) The
dataset is licensed under a broad set of terms of use which can be found
at https://ai.facebook.com/datasets/segment-anything. (4) The data is more
geographically diverse than its predecessors, and we hope it will bring the
community one step closer to creating fairer and more equitable models.

2. Who created the dataset (e.g., which team, research group) and on behalf
of which entity (e.g., company, institution, organization)? The dataset was
created by the FAIR team of Meta AI. The underlying images were collected
and licensed from a third party photo company.

3. Who funded the creation of the dataset? If there is an associated grant,
please provide the name of the grantor and the grant name and number.
Meta AI funded the creation of the dataset.

4. Any other comments? No.

Composition

1. What do the instances that comprise the dataset represent (e.g., documents,
photos, people, countries)? Are there multiple types of instances (e.g.,
movies, users, and ratings; people and interactions between them; nodes
and edges)? Please provide a description. All of the instances in the dataset
are photos. The photos vary in subject matter; common themes of the photo
include: locations, objects, scenes. All of the photos are distinct, however
there are some sets of photos that were taken of the same subject matter.

2. How many instances are there in total (of each type, if appropriate)? There

are 11 million images.

3. Does the dataset contain all possible instances or is it a sample (not nec-
essarily random) of instances from a larger set? If the dataset is a sample,
then what is the larger set? Is the sample representative of the larger set
(e.g., geographic coverage)? If so, please describe how this representa-
tiveness was validated/veriﬁed. If it is not representative of the larger set,
please describe why not (e.g., to cover a more diverse range of instances,
because instances were withheld or unavailable). The dataset is composed
of images licensed from a photo provider. The dataset contains all instances
licensed. The images are photos, i.e. not artwork, although there are a few
exceptions. The dataset includes all generated masks for each image in the
dataset. We withheld ∼2k randomly selected images for testing purposes.

4. What data does each instance consist of? “Raw” data (e.g., unprocessed
text or images) or features? In either case, please provide a description.
Each instance in the dataset is an image. The images were processed to blur
faces and license plates to protect the identities of those in the image.

5.

6.

Is there a label or target associated with each instance? If so, please provide
a description. Each image is annotated with masks. There are no categories
or text associated with the masks. The average image has ∼100 masks, and
there are ∼1.1B masks in total.

Is any information missing from individual instances? If so, please provide
a description, explaining why this information is missing (e.g., because it
was unavailable). This does not include intentionally removed information,
but might include, e.g., redacted text. Yes. Each image is accompanied by
a short caption that describes the content and place of the photo in a free
form text. Per our agreement with the photo provider we are not allowed to
release these captions. However, we use them in our paper to analyze the
geographical distribution of the dataset.

7. Are relationships between individual instances made explicit (e.g., users’
movie ratings, social network links)? If so, please describe how these rela-
tionships are made explicit. No, there are no known relationships between
instances in the dataset.

8. Are there any errors, sources of noise, or redundancies in the dataset? If
so, please provide a description. Errors: The masks are generated by a
segmentation model, so there may be errors or inconsistencies in the masks.
Redundancies: While no two images are the same, there are instances of
images of the same subject taken close together in time.

9.

Is the dataset self-contained, or does it link to or otherwise rely on external
resources (e.g., websites, tweets, other datasets)? If it links to or relies on
external resources, a) are there guarantees that they will exist, and remain
constant, over time; b) are there ofﬁcial archival versions of the complete
dataset (i.e., including the external resources as they existed at the time
the dataset was created); c) are there any restrictions (e.g., licenses, fees)
associated with any of the external resources that might apply to a dataset
consumer? Please provide descriptions of all external resources and any
restrictions associated with them, as well as links or other access points, as
appropriate. The dataset is self-contained.

10. Does the dataset contain data that might be considered conﬁdential (e.g.,
data that is protected by legal privilege or by doctor-patient conﬁdentiality,
data that includes the content of individuals’ non-public communications)?
If so, please provide a description. No.

11. Does the dataset contain data that, if viewed directly, might be offensive,
insulting, threatening, or might otherwise cause anxiety? If so, please de-
scribe why. We have two safety measures to prevent objectionable content:
(1) Photos are licensed from a photo provider and had to meet the terms of
service of the photo provider. We requested that all objectionable content
be ﬁltered from the images we licensed. (2) If a user observes objectionable
image(s) in the dataset, we invite them to report the image(s) at segment-
anything@meta.com for removal. Despite the measures taken, we observe
that a small portion of images contains scenes of protests or other gatherings
that focus on a diverse spectrum of religious beliefs or political opinions that
may be offensive. We were not able to produce a ﬁltering strategy that re-
moves all such images and rely on users to report this type of content.

12. Does the dataset identify any subpopulations (e.g., by age, gender)? If so,
please describe how these subpopulations are identiﬁed and provide a de-
scription of their respective distributions within the dataset. The dataset
does not identify any subpopulations of the people in the photos.

13.

Is it possible to identify individuals (i.e., one or more natural persons), ei-
ther directly or indirectly (i.e., in combination with other data) from the
dataset? If so, please describe how. No. Images were subjected to a face
blurring model to remove any personally identiﬁable information. If a user
observes any anonymization issue, we invite them to report the issue and
the image id(s) at segment-anything@meta.com.

14. Does the dataset contain data that might be considered sensitive in any way
(e.g., data that reveals race or ethnic origins, sexual orientations, religious
beliefs, political opinions or union memberships, or locations; ﬁnancial or
health data; biometric or genetic data; forms of government identiﬁcation,
such as social security numbers; criminal history)? If so, please provide
a description. The dataset contains scenes of protests, or other gatherings
that may suggest religious beliefs, political opinions or union memberships.
However, the faces of all people in the dataset have been anonymized via
facial blurring, so it is not possible to identify any person in the dataset.

15. Any other comments? No.

Collection Process

1. How was the data associated with each instance acquired? Was the data
directly observable (e.g., raw text, movie ratings), reported by subjects (e.g.,
survey responses), or indirectly inferred/derived from other data (e.g., part-
of-speech tags, model-based guesses for age or language)? If the data was
reported by subjects or indirectly inferred/derived from other data, was the
data validated/veriﬁed? If so, please describe how. The released masks
associated with each image were automatically inferred by our segmentation
model, SAM. The masks that were collected using model-assisted manual
annotation will not be released. Quality was validated as described in §5.

2. What mechanisms or procedures were used to collect the data (e.g., hard-
ware apparatuses or sensors, manual human curation, software programs,
software APIs)? How were these mechanisms or procedures validated? The
images in the dataset are licensed from an image provider. They are all pho-
tos taken by photographers with different cameras.

25

3.

If the dataset is a sample from a larger set, what was the sampling strategy
(e.g., deterministic, probabilistic with speciﬁc sampling probabilities)? We
withheld ∼2k randomly selected images for testing purposes. The rest of
the licensed images are included in the dataset.

4. Who was involved in the data collection process (e.g., students, crowdwork-
ers, contractors) and how were they compensated (e.g., how much were
crowdworkers paid)? The released masks were automatically inferred by
SAM. For details on our model-assisted manual annotation process see our
Data Annotation Card in §F.2. Note these masks will not be released.

5. Over what timeframe was the data collected? Does this timeframe match
the creation timeframe of the data associated with the instances (e.g., recent
crawl of old news articles)? If not, please describe the timeframe in which
the data associated with the instances was created. The licensed photos
vary in their date taken over a wide range of years up to 2022.

6. Were any ethical review processes conducted (e.g., by an institutional re-
view board)? If so, please provide a description of these review processes,
including the outcomes, as well as a link or other access point to any sup-
porting documentation. If the dataset does not relate to people, you may skip
the remaining questions in this section. We underwent an internal privacy
review to evaluate and determine how to mitigate any potential risks with
respect to the privacy of people in the photos. Blurring faces and license
plates protects the privacy of the people in the photos.

7. Did you collect the data from the individuals in question directly, or obtain
it via third parties or other sources (e.g., websites)? We licensed the data
from a third party photo provider.

8. Were the individuals in question notiﬁed about the data collection? If so,
please describe (or show with screenshots or other information) how no-
tice was provided, and provide a link or other access point to, or other-
wise reproduce, the exact language of the notiﬁcation itself. The images
are licensed from a third party who provided appropriate representations
regarding the collection of any notices and consents as required from indi-
viduals. In addition, all identiﬁable information (e.g. faces, license plates)
was blurred. Under the terms of the dataset license it is prohibited to attempt
to identify or associate an image with a particular individual.

9. Did the individuals in question consent to the collection and use of their
data? If so, please describe (or show with screenshots or other informa-
tion) how consent was requested and provided, and provide a link or other
access point to, or otherwise reproduce, the exact language to which the
individuals consented. The images are licensed from a third party who pro-
vided appropriate representations regarding the collection of any notices and
consents as required from individuals. In addition, all identiﬁable informa-
tion (e.g. faces, license plates) was blurred from all images. For avoidance
of doubt, under the terms of the dataset license it is prohibited to attempt to
identify or associate an image with a particular individual.

10.

If consent was obtained, were the consenting individuals provided with a
mechanism to revoke their consent in the future or for certain uses? If
so, please provide a description, as well as a link or other access point
to the mechanism (if appropriate). We invite users to report at segment-
anything@meta.com for image(s) removal.

11. Has an analysis of the potential impact of the dataset and its use on data
subjects (e.g., a data protection impact analysis) been conducted? If so,
please provide a description of this analysis, including the outcomes, as
well as a link or other access point to any supporting documentation. To
eliminate any potential impact on people whose photos are included in the
dataset, identiﬁable information (faces, license plates) has been blurred.

12. Any other comments? No.

Preprocessing / Cleaning / Labeling
1. Was any preprocessing / cleaning / labeling of the data done (e.g., dis-
cretization or bucketing, tokenization, part-of-speech tagging, SIFT fea-
ture extraction, removal of instances, processing of missing values)? If so,
please provide a description. If not, you may skip the remaining questions
in this section. We resized the high-resolution licensed images such that
the shorter side is 1500 pixels and only processed the images to remove any
identiﬁable and personal information from the photos (faces, license plates).

2. Was the “raw” data saved in addition to the preprocessed/cleaned/labeled
data (e.g., to support unanticipated future uses)? If so, please provide a link
or other access point to the “raw” data. No, as we removed the data for
safety reasons and to respect privacy, we do not release the unaltered photos.

3.

Is the software that was used to preprocess/clean/label the data avail-
able? If so, please provide a link or other access point. We used the

RetinaFace [88, 89] model (https://github.com/serengil/retinaface) to detect
faces. The model used to blur license plates has not been made public.

Uses
1. Has the dataset been used for any tasks already? If so, please provide a
description. The dataset was used to train our segmentation model, SAM.

2.

Is there a repository that links to any or all papers or systems that use the
dataset? If so, please provide a link or other access point. No. However, all
users of the dataset must cite it, so its use is trackable via citation explorers.

3. What (other) tasks could the dataset be used for? We intend the dataset
to be a large-scale segmentation dataset. However, we invite the research
community to gather additional annotations for the dataset.

4.

Is there anything about the composition of the dataset or the way it was
collected and preprocessed/cleaned/labeled that might impact future uses?
For example, is there anything that a dataset consumer might need to know
to avoid uses that could result in unfair treatment of individuals or groups
(e.g., stereotyping, quality of service issues) or other risks or harms (e.g.,
legal risks, ﬁnancial harms)? If so, please provide a description. Is there
anything a dataset consumer could do to mitigate these risks or harms? We
have an analysis of the approximate geographic and income level coverage
of our dataset in §6. While we believe our dataset to be more representative
than most of the publicly existing datasets at this time, we acknowledge
that we do not have parity across all groups, and we encourage users to be
mindful of potential biases their models have learned using this dataset.

5. Are there tasks for which the dataset should not be used? If so, please pro-
vide a description. Full terms of use for the dataset including prohibited use
cases can be found at https://ai.facebook.com/datasets/segment-anything.

6. Any other comments? No.

Distribution
1. Will the dataset be distributed to third parties outside of the entity (e.g.,
company, institution, organization) on behalf of which the dataset was cre-
ated? If so, please provide a description. The dataset will be available for
the research community.

2. How will the dataset will be distributed (e.g., tarball on website, API,
GitHub)? Does the dataset have a digital object identiﬁer (DOI)? The
dataset is available at https://ai.facebook.com/datasets/segment-anything.

3. When will the dataset be distributed? The dataset will be released in 2023.

4. Will the dataset be distributed under a copyright or other intellectual
property (IP) license, and/or under applicable terms of use (ToU)? If
so, please describe this license and/or ToU, and provide a link or other
access point to, or otherwise reproduce, any relevant licensing terms
or ToU, as well as any fees associated with these restrictions. Yes.
The license agreement and terms of use for the dataset can be found at
https://ai.facebook.com/datasets/segment-anything. Users must agree to the
terms of use before downloading or using the dataset.

5. Have any third parties imposed IP-based or other restrictions on the data
associated with the instances? If so, please describe these restrictions, and
provide a link or other access point to, or otherwise reproduce, any relevant
licensing terms, as well as any fees associated with these restrictions. Full
terms of use and restrictions on use of the SA-1B dataset can be found at
https://ai.facebook.com/datasets/segment-anything.

6. Do any export controls or other regulatory restrictions apply to the dataset
or to individual instances? If so, please describe these restrictions, and pro-
vide a link or other access point to, or otherwise reproduce, any supporting
documentation. The license and restrictions on use of the SA-1B dataset
can be found at https://ai.facebook.com/datasets/segment-anything.

7. Any other comments? No.

Maintenance
1. Who will be supporting/hosting/maintaining the dataset? The dataset will
be hosted at https://ai.facebook.com/datasets/segment-anything and main-
tained by Meta AI.

2. How can the owner/curator/manager of the dataset be contacted (e.g., email

address)? Please email segment-anything@meta.com.

3.

Is there an erratum? If so, please provide a link or other access point. No.

4. Will the dataset be updated (e.g., to correct labeling errors, add new in-
stances, delete instances)? If so, please describe how often, by whom, and
how updates will be communicated to dataset consumers (e.g., mailing list,

26

GitHub)? To aid reproducibility of research using SA-1B, the only updates
will be to remove reported images.

3. Were sociodemographic characteristics used to select annotators for your

task? If so, please detail the process. No.

5.

If the dataset relates to people, are there applicable limits on the retention of
the data associated with the instances (e.g., were the individuals in question
told that their data would be retained for a ﬁxed period of time and then
deleted)? If so, please describe these limits and explain how they will be
enforced. There are no limits on data retention. We took measures to remove
personally identiﬁable information from any images of people. Users may
report content for potential removal here: segment-anything@meta.com.

6. Will

of

the

older

versions

to
sup-
ported/hosted/maintained?
If not, please
describe how its obsolescence will be communicated to dataset consumers.
No, as the only updates will be to remove potentially harmful content, we
will not keep older versions with the content.

continue
If so, please describe how.

dataset

be

7.

If others want to extend/augment/build on/contribute to the dataset, is there
a mechanism for them to do so? If so, please provide a description. Will
these contributions be validated/veriﬁed? If so, please describe how. If not,
why not? Is there a process for communicating/distributing these contribu-
tions to dataset consumers? If so, please provide a description. We encour-
age users to gather further annotations for SA-1B. Any users who generate
annotations will be liable for hosting and distributing their annotations.

8. Any other comments? No.

F.2. Data Annotation Card

Task Formulation

1. At a high level, what are the subjective aspects of your task? Segmenting
objects present in an image is inherently a subjective task. For instance,
one annotator may segment two boots as one mask, whereas another may
segment each boot separately. Depending on annotators’s skills, the quality
of the mask and the number of masks per image are different between an-
notators. Despite these subjective aspects of the task, we believed efﬁcient
annotation was possible as the data was annotated in a per-mask fashion
with the main focus on the diversity of the data rather than completeness.

2. What assumptions do you make about annotators? Our annotators worked
full time on our annotation task with very small attrition rate. This made
it possible to train the annotators providing feedback and answering their
questions on a regular basis. Speciﬁcally: (1) By giving a clear understand-
ing of the goals of this work and providing clear guidelines, including vi-
suals and video recordings of the tasks, annotators had enough context to
understand and perform the tasks reasonably. (2) Sharing objectives and
key results and meeting weekly with annotators increased the likelihood
that annotators improved annotation quality and quantity over time.

3. How did you choose the speciﬁc wording of your task instructions? What
steps, if any, were taken to verify the clarity of task instructions and wording
for annotators? As our task was annotating images, the annotation guide-
lines included visual examples. Our research team completed 30 annotation
tasks to identify any obvious challenges using the annotation tool, collec-
tively decide how to handle complex cases, and reﬁne the guidelines. The
research team met with the annotators weekly for feedback sessions. Videos
of the research team performing the task were shared live with the annota-
tors, followed by Q&A sessions. Annotators were able to give feedback on
unclear aspects, both during the feedback session and asynchronously.

4. What, if any, risks did your task pose for annotators and were they informed
of the risks prior to engagement with the task? No identiﬁed risks. Images
were ﬁltered for objectionable content prior to the annotation phase.

5. What are the precise instructions that were provided to annotators? We
provide only high-level instructions: Given an image, we aim at segment-
ing every possible object. Annotators generate a mask for every potential
object they can identify. An object can be segmented using our interactive
segmentation tool either by using corrective foreground/background clicks
to add/remove parts of the mask or by drawing a bounding box around the
object. Masks can be reﬁned using pixel-precise tools.

Selecting Annotations

1. Are there certain perspectives that should be privileged? If so, how did you
seek these perspectives out? We chose to work with annotators that have
worked on other vision annotation tasks before.

2. Are there certain perspectives that would be harmful to include? If so, how

did you screen these perspectives out? No.

27

4.

If you have any aggregated socio-demographic statistics about your anno-
tator pool, please describe. Do you have reason to believe that sociode-
mographic characteristics of annotators may have impacted how they an-
notated the data? Why or why not? We worked with 130 annotators. The
annotators were all based in Kenya. We do not believe sociodemographic
characteristics of annotators meaningfully impacted the annotated data.

5. Consider the intended context of use of the dataset and the individuals
and communities that may be impacted by a model trained on this dataset.
Are these communities represented in your annotator pool? The Segment
Anything 1B (SA-1B) dataset is to be used for research purposes only.
The SA-1B dataset is one of the most geographically diverse segmentation
dataset, as discussed in §6. In addition, we analyze the responsible AI axes
of a model trained on the dataset in §6.

Platform and Infrastructure Choices
1. What annotation platform did you utilize? At a high level, what considera-
tions informed your decision to choose this platform? Did the chosen plat-
form sufﬁciently meet the requirements you outlined for annotator pools?
Are any aspects not covered? We used a proprietary annotation platform.

2. What, if any, communication channels did your chosen platform offer to
facilitate communication with annotators? How did this channel of com-
munication inﬂuence the annotation process and/or resulting annotations?
We manually reviewed annotations and shared feedback with the annotators
on a weekly basis. We communicated common mistakes or inconsisten-
cies and the corresponding corrections.
In addition, the annotators were
given feedback for improvements daily by the annotation QA team. Out-
side the weekly feedback sessions, annotators had access to a spreadsheet
and chat group to facilitate communication with the research team. This
process greatly improved the average speed and quality of the annotations.

3. How much were annotators compensated? Did you consider any partic-
ular pay standards, when determining their compensation? If so, please
describe. Annotators were compensated with an hourly wage set by the
vendor. The vendor is a Certiﬁed B Corporation.

Dataset Analysis and Evaluation
1. How do you deﬁne the quality of annotations in your context, and how did
you assess the quality in the dataset you constructed? Annotators were ﬁrst
placed into training. They followed a 1-day training session led by the ven-
dor and then were asked to annotate a large number of examples from a
training queue. Annotators graduated from training to production after the
vendor QA team, in collaboration with the research team, manually spot-
checked the annotator’s masks to ensure quality. On average, annotators
spent one week in training before graduating. Production quality assess-
ment followed a similar process: the vendor QA team and the research team
manually reviewed the annotations weekly, sharing feedback weekly.

2. Have you conducted any analysis on disagreement patterns? If so, what
analyses did you use and what were the major ﬁndings? Did you analyze
potential sources of disagreement? We pointed out common mistakes dur-
ing weekly meetings with the annotators.

3. How do the individual annotator responses relate to the ﬁnal labels released
in the dataset? The annotations were only used to train early versions of the
SAM model and we do not currently plan to release them.

Dataset Release and Maintenance
1. Do you have reason to believe the annotations in this dataset may change
over time? Do you plan to update your dataset? No, except to remove
objectionable images.

2. Are there any conditions or deﬁnitions that, if changed, could impact the

utility of your dataset? We do not believe so.

3. Will you attempt to track, impose limitations on, or otherwise inﬂuence how
your dataset is used? If so, how? The SA-1B dataset will be released under
a license agreement allowing use for certain research purposes and protec-
tions for researchers. Researchers must agree to the terms of the license
agreement to access the dataset.

4. Were annotators informed about how the data is externalized? If changes to
the dataset are made, will they be informed? No, we do not plan to release
the manual annotations at the moment.

5.

Is there a process by which annotators can later choose to withdraw their
data from the dataset? If so, please detail. No.

Model Overview

Name
Version
Date
Organization
Mode type
Architecture
Repository
Citation
License

SAM or Segment Anything Model
1.0
2023
The FAIR team of Meta AI
Promptable segmentation model
See §3
https://github.com/facebookresearch/segment-anything
https://research.facebook.com/publications/segment-anything
Apache 2.0

Intended Use

Primary intended uses

Primary intended users

Out-of-scope use cases

Caveats and recommendations

primarily

developed

SAM is intended to be used for any prompt-based segmentation task. We explored its use in segmenting objects
from a point (§7.1), edge detection (§7.2), segmenting all objects (§7.3), and segmenting detected objects (§7.4).
We explored how SAM can integrate with other vision models to segment objects from text (§7.5).
SAM was
research.
https://github.com/facebookresearch/segment-anything.
See terms of use for SAM found at https://github.com/facebookresearch/segment-anything. See Use Cases under
Ethical Considerations.
SAM has impressive zero-shot performance across a wide range of tasks. We note, however, that in the zero-shot
setting there may be multiple valid ground truth masks for a given input. We recommend users take this into
consideration when using SAM for zero-shot segmentation. SAM can miss ﬁne structures and can hallucinate
small disconnected components. See §8 for a discussion of limitations.

for SAM can

license

found

The

for

be

at

Relevant Factors

Groups

SAM was designed to segment any object. This includes stuff and things.

Instrumentation and environment We benchmarked SAM on a diverse set of datasets and found that SAM can handle a variety of visual data including
simulations, paintings, underwater images, microscopy images, driving data, stereo images, ﬁsh-eye images. See
§D.1 and Table 7 for information on the benchmarks used.

Metrics

Model performance measures We evaluated SAM on a variety of metrics based on the downstream task in our experiments.

• mIoU: We used the mean intersection-over-union after a given number of prompts to evaluate the segmen-

tation quality of a mask when prompted with points.

• Human evaluation: We performed a human study (detailed in §E) to evaluate the real world performance
of SAM. We compared the masks generated by SAM to a baseline state-of-the-art interactive segmentation
model, RITM [92], using a perceptual quality scale from 1 to 10.

• AP: We used average precision to evaluate instance segmentation for a given box and edge detection.
• AR@1000: We used average recall to evaluate object proposal generation.
• ODS, OIS, AP, R50: We used the standard edge detection evaluation metrics from BSDS500 [72, 3].

Evaluation Data

Training Data

Data sources

See §D.1.

Data source

See Data Card in §F.1.

Ethical Considerations

Cost and impact of compute

Data We trained SAM on licensed images. The images were ﬁltered for objectionable content by the provider, but we
acknowledge the possibility of false negatives. We performed a geographic analysis of the SA-1B dataset in §6.
While SA-1B is more geographically diverse than many of its predecessors, we acknowledge that some geographic
regions and economic groups are underrepresented.
SAM was trained on 256 A100 GPUS for 68 hours. We acknowledge the environmental impact and cost of training
large scale models. The environmental impact of training the released SAM model is approximately 6963 kWh
resulting in an estimated 2.8 metric tons of carbon dioxide given the speciﬁc data center used, using the calculation
described in [77] and the ML CO2 Impact calculator [61]. This is equivalent to ∼7k miles driven by the average
gasoline-powered passenger vehicle in the US [101]. We released the SAM models to both reduce the need for
retraining and lower the barrier to entry for large scale vision research.

Risks and harms We evaluated SAM for fairness in §6. Downstream use cases of SAM will create their own potential for biases
and fairness concerns. As such we recommend users run their own fairness evaluation when using SAM for their
speciﬁc use case.

Use cases We implore users to use their best judgement for downstream use of the model.

Table 9: Model Card for SAM, following the procedure detailed in [75].

28

•

•

–

We have several models that, when provided with a click or a box as input, output a mask. We would
like to compare the quality of these models by rating the quality of their masks on many examples.
The interface will be different than for regular mask annotation.

Each job reviews one mask in one image.

On the right, there will be ﬁve image thumbnails in two rows. Each thumbnail can be moused-
over to show the image at a larger size. Clicking on the thumbnail will make it full screen, and
clicking again will return to the original screen.

The images show the same mask in ﬁve different views. On the top row: (left) the image
without the mask, (middle) the mask overlaid on the image, and (right) the mask alone. On
the bottom row: (left) a zoomed in view of the object without a mask, and (right) a zoomed
in view of the mask overlaid on the image. These views are provided to make it easy to see
different types of mask errors.
The mask will be in red when overlaid on the image.

–
– When shown by itself, the mask is yellow, and the background is purple.
–

Each image will include either a blue dot or a blue and white box. This is the input to the
model, as if you had clicked at this location or drawn this box.

•

On the left, there are buttons labeled 1-10. This is used to rate the quality of the shown mask.

Objective and Setup

Example interface page. There will be ﬁve images on the
right and a question box on the left.

Mouse over an image to show the full image.

Click on an image to make it full screen. The arrows will cy-
cle between images. Click again to return to previous view.

The ﬁrst image on the top row shows the image without a
mask. A blue point will be on the object of interest, or a
blue and white box will surround it.

The second image on the top row shows the mask for the
object in red.

The third image on the top row shows the mask only. The
mask is in yellow and the background is purple.

The ﬁrst image on the bottom row shows a zoomed in view
of the object without a mask.

Does the mask correspond to an actual object?

•

Valid objects can include:

What we would like you to do for each job:

•

Please aim to spend up to 30 seconds per job.

• Mouse-over or click each of the three images of the mask on the right to get a sense of the
quality of the mask. The thumbnail is too small to judge a mask, do not judge a mask by the
thumbnail alone. Each image can provide a different signal on possible mask errors:

–

–
–

–
–
–

•

•

•

The unzoomed image can give context for the mask: does this mask correspond to an actual
object?
The mask-only image can show if the mask has small holes or separated, incorrect pixels.
The zoomed image can show if the mask boundaries make sense.

Judge the quality of the mask on three criterion. Examples will follow.

Does the mask correspond to an actual object?
Does the mask have a good boundary?
Does the mask correspond to the provided point or box?

Rate the quality of the mask on a scale of 1-10 using the drop-down box on the left.

Next are details and examples for judging mask quality according to the three criterion. These
are just examples and other cases may come up, please use your best judgment when deter-
mining if something is a good mask.

•

•

–
–
–
–

–

Entire single objects (such as a person, shirt, or tree)
Logical parts of objects (a chair leg, a car door, a tabletop)
Collections of objects (a stack of books, a crowd of people)
‘Stuff’ (the ground, the sky).

Example errors a mask may have. The severity of these errors may be minor or major:

Include a piece of another object (the mask of a person including the arm of a nearby
person)

– Miss part of an object (the mask covers only one part of a building obscured by a tree in

–
–

the foreground),
Combine two unrelated things (a single mask covers both a mug and a pen on a desk)
Include an arbitrary part of a collection for a point input (a point is on one apple, but
the mask covers three apples in a pile of many apples). If a box surrounds an arbitrary
collection, it is not an error to provide a mask for these objects.

If you are unsure, a good rule-of-thumb is: can you name the object in question? However,
some things that are hard to name may still be good objects (an unusual component of a
machine, something at the edge of the image for which it is hard to determine what it is).

The second image on the bottom row shows a zoomed in
view of the object with a mask. The mask is in red.

On the left are buttons to rate the mask quality, with selec-
tions 1-10.

Task

Judging Mask Quality (1 of 3)

Does the mask have a good boundary?

•

Errors in the boundary can include:

–
–
–
–

–

Incorrect holes in the mask
Incorrect pixels included separated from the main part of the mask
Poor edge quality, where the mask does not exactly match the edge of the object.
Failure to consistently handle obscuring foreground objects (a mask that covers obscuring
objects is ﬁne, and a mask that doesn’t cover obscuring objects is ﬁne, but one that does
some of both has an error)
Pixelation of a small mask is not an error, as long as the mask still matches the edges of
the object.

Does the mask correspond to the provided point or box?

•

For points:

–
–

The point needs to be on the mask.
The size or position of the object with respect to the point does not matter (a point on
someone’s gloved hand can correspond to the glove or to the entire person, both are valid
masks).

•

For boxes:

–

–

The object needs to be the best object that is the size of the box (if a box is around some-
one’s entire head but the mask is of their hair, this is an error: their hair is in the box but is
not the correct object).
If the box clearly corresponds to a given object but is slightly smaller than it, it is okay if
the mask goes slightly outside a box (if a box around a person misses their extended hand,
the mask can still include their hand even if the mask goes outside the box).

Judging Mask Quality (2 of 3)

Judging Mask Quality (3 of 3)

Example error of ‘Include a piece of another object’: The
elephant mask contains a piece of another nearby elephant.

Example error of ‘Missing a part of an object’: the mask is
missing a disconnected part of the object: the back half of
the zebra, and the right portion of the plate.

Example error of ‘Include an arbitrary part of a collection’:
In top top image, the point is on one orange rind, but the
mask covers two orange rinds. This is a mask error:
the
mask covers an arbitrary number of objects in the collection,
and should either cover one orange rind or all of them. In
the bottom image, the box is around both vegetables. Since
this is the best match to the box, this is not a mask error.

Example error for ‘Incorrect holes in the mask’: This mask
has holes in the upper left and on the left sides (black ar-
rows). These holes are much easier to see on the ‘mask
only’ image.

Example error for ‘Incorrect pixels included separated from
the main part of the mask’: The ‘mask only’ view reveals a
few stray incorrect pixels on the clock face.

Example error for ‘Poor edge quality’: The mask has poor
edge quality, both along the edge of the umbrella, as well as
along the thin pole.

Figure 19: Here we provide the complete guidelines given to annotations for the human review of mask quality. Some images
been edited slightly and faces have been blurred to enable release. Best viewed with zoom (part 1 of 2).

G. Annotation Guidelines

We provide the complete guidelines given to annotations
for the human review of mask quality in Fig. 19 and Fig. 20.

29

Example for ‘Combine two unrelated things’: The point in-
dicates the lizard, but the mask covers both the lizard and a
bird. This is a mask error.

Example error for ‘Failure to consistently handle obscuring
foreground objects’: The pole on the right (blue arrow) is
excluded from the mask, while the pole on the left is in-
cluded in the object (black arrow). The mask should either
include or exclude both of these.

Example of ‘Pixelation of a small mask’: this mask has an
imperfect boundary, since it extends beyond the object at
the black arrow. However, the ‘blocky’ pattern of the mask
is not an error, since, when zoomed in this much, the image
is also blocky the same way.

Example error for consistency with the provided point: The
mask does not agree with the blue point, so this is a mask
error.

Example for consistency with the provided point: For this
input point, but the logo (left) and the container (right) are
valid objects, since the blue point lies on both of them. Nei-
ther mask has a mask error.

Example for consistency with a box: The box surrounds the
bowl of oranges, but the mask is only of a single orange.
This is a mask error.

Example for consistency with a box: The box’s shape ﬁts
the zebra. Even though the mask extends slightly outside
the box to include the zebra’s left leg, this is not an error.

Mask Scoring

Overall mask quality is subjective, each of the above errors may hurt mask quality only a little or a
lot, depending on how large the error is. Please use your best judgment when choosing mask scores,
and try to stay consistent from mask-to-mask. Here are some general guidelines for what different
scores should correspond to:

•

•

•

•

•

A score of 1: It is not possible to tell what object this mask corresponds to. This includes the
case that there is no mask visible at all.

A low score (2-4): The object is mostly identiﬁable, but the mask quality is extremely poor
(e.g. large regions of the mask cover other objects; large regions of the object missing; ex-
tremely splotchy mask boundaries that cut through the middle of the object).

A mid score (5-6): The object is identiﬁable and the boundary is mostly correct, but there
are major errors (missing a signiﬁcant disconnected part of the object; containing a signiﬁcant
part of another object; very poor boundary quality in one area of the object but not the entire
object).

A high score (7-9): The object is identiﬁable and errors are small and rare (missing a small,
heavily obscured disconnected component, having small regions where the mask boundary
does not quite match the object boundary).

A score of 10: The mask is pixel-perfect; it has no identiﬁable errors at all.

Example of a mask with a score of 1: It is not clear what
object this mask corresponds to.

Example of a mask with a low score (2-4): The main ob-
ject is identiﬁable, but the mask includes a large, incorrect
portion of another object.

Example of a mask with a low score (2-4): The main ob-
ject is identiﬁable, but a large, random part of the object is
missing.

Example of a mask with a low-to-medium score (4-5): The
object is identiﬁable and the edges are all correct, but the
mask incorrectly includes the hand of the person on the left.

Example of a mask with a medium score (5-6): The mask
clearly corresponds to the plate, but the boundary with the
wafﬂe is quite poor.

Example of a mask with a medium score (5-6): the object
is easy to identify, and most of the edges make sense. How-
ever, there is a signiﬁcant disconnected part (their arm inside
the frame) that is mostly missing, as well as splotchy pixels
in this region.

Example of a mask with a medium-to-high score (6-8): the
mask has two small-ish regions of poor boundary, at the top
of the mask and on the bottom right.

Example of a mask with a medium-to-high score (6-8): The
wreath is a valid object that is the size of the box (the entire
wreath + clock would also be a valid object). However, there
are incorrect stray mask pixels on the clock.

Example of a mask with a high score (7-9): The boundary of
the horse is almost entirely correct, except for the right side
of its back leg. The mask consistently includes all of the
equipment that horse is wearing, and has logical boundaries.

Example of a mask with a very high score (∼9): There are
only minor errors around the edge of the mask. The blocky
‘pixelation’ is not an error, since the image is also blocky at
this scale.

Example of a mask with a very high score (9-10): the mask
has only very minor errors in the edge on the bottom right.

Example of a mask with a very high score (9-10): There are
only minor errors around the edge of the mask.

Figure 20: Here we provide the complete guidelines given to annotations for the human review of mask quality. Some images
been edited slightly and faces have been blurred to enable release. Best viewed with zoom (part 2 of 2).

30

