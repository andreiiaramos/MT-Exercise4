# MT Exercise 4 - Results


### Issues in the code
We encountered three obstacles during the Initial run of the pipeline. 
> First, a strict naming convention check introduced in the latest Hugging Face release failed to parse a specific edge case in our dataset schema. To bypass this asset-loading failure, the library was temporarily downgraded to an earlier, more permissive version to unblock the download pipeline.

> Second, executing the POSIX-compliant (Mac/Linux) bash scripts via Git Bash on a Windows 11 host created environment pathing and executable resolution conflicts. Because Windows typically maps the Python interpreter to python.exe while the scripts explicitly call python3, a symlink was injected, by us, into the Python installation directory to map all python3 invocations directly to the local python binary.

> Lastly refactoring  virtualenv to venv: Due to persistent compatibility and execution issues when initializing the environment via the external virtualenv package on our devices, we refactored make_virtualenv.sh to use Python’s built-in venv module instead.

## Part 1 Experiments with Byte Pair Encoding

The following experiment was conducred on the English -> Italian translation direction from the IWSLT 2017 data set. In total, three models were trained and evaluated for their performance.

The first model used the word_level approach with a vocabulary limited to 2000 items.
The second and third models relied on BPE-subwords; the former utilized 2000-item vocabulary while the latter worked with 8000 items.

Here are the results for the first part:

|Model   | BPE | Vocabulary Size | BLEU Score|
|--------| ----|-----------------| ----------|
|word_2k | no  | 2000            | 11.2      |
|bpe_2k  | yes | 2000            | 20.2      |
|bpe_8k  | yes | 8000            | 21.9      |


The first model managed to obtain the BLEU score of only 11.2, as the limited vocabulary was not enough to tackle infrequent words effectively, and cause substantial out-of-vocabulary issues and reduced translation quality.

At the same time, the BLEU result for the BPE model working with a 2000 word dictionary amounted to 20.2, greatly outperforming the previous model and proving the importance of subword segmentation in NMT.

Furthermore, the increase in the size of the subword vocabulary from 2000 to 8000 positively affected the model's performance, resulting in a BLEU score of 21.9. It can be concluded that the larger vocabulary size was responsible for the generation of more proper segmentation and less word splitting.

All these experiments prove that BPE vocabularies work far better than word vocabularies when performing English -> Italian translation using neural machine learning.

Here is an example:

| Type | Translation |
|---|---|
| Source (EN) | Last year I showed these two slides so that demonstrate that the arctic ice cap, which for most of the last three million years has been the size of the lower 48 states, has shrunk by 40 percent. |
| Gold Reference (IT) | L'anno scorso ho mostrato queste diapositive per dimostrare che la calotta glaciale artica, che per quasi tre milioni di anni ha avuto le dimensioni dei 48 Stati Uniti continentali, si è ristretta del 40%. |
| word_2k | L' anno scorso ho mostrato queste due `<unk>` `<unk>` che il ghiaccio `<unk>` `<unk>`, che per la maggior parte degli ultimi tre milioni di anni sono stati `<unk>` la dimensione della `<unk>` `<unk>`, ha `<unk>` `<unk>` dal 40 %. |
| bpe_2k | L' anno scorso anno ho mostrato questi due strumento così che la cap artico che la cap artico, che per la maggior parte dei ultimi tre milioni di anni è stato la dimensione del 48 stati, ha condiviso da 48 %. |
| bpe_8k | L' anno scorso ho mostrato queste due diapositive così che dimostrano che il ghiaccio si ha fatto che il ghiaccio si è rivelato che per la maggior parte degli ultimi tre milioni di anni sono stati più bassa 48 stati, ha sciato di 40 %. |

This example shows the limitations of the word_2k model: its restricted size of 2000 left many key content words as `<unk>` tokens, which made the translations much more difficult to comprehend, and led to a significant amount of loss of information. Conversely, both BPE-based models were able to do this without replacing anything with unknown tokens since they can break down infrequent words into smaller subword units and generate translations that maintain a much greater degree of meaning from the source than by using a word-level model. 
The BPE_8k model produced the translation that was closest to the gold reference by again translating "slides" as "diapositive" and retaining more of the original sentnece structure. This pattern is consistent with what is seen when measuring BLEU values, and in fact the BPE systems have a significantly higher BLEU score.

## Part 2 Impact of beam size on translation quality


Here are the results of the second part:

| Beam Size | BLEU Score | Translation Time (s) |
|------------|------------|----------------------|
| 1          | 19.9       | 19                   |
| 2          | 21.7       | 20                   |
| 3          | 21.9       | 23                   |
| 4          | 22.0       | 25                   |
| 5          | 21.9       | 27                   |
| 6          | 21.9       | 33                   |
| 7          | 21.9       | 37                   |
| 8          | 22.0       | 41                   |
| 9          | 22.1       | 46                   |
| 10         | 21.9       | 51                   |


In the beam search experiment results, it is found that an increase in the beam size greatly improved the translation performance than using greedy decoding. The BLEU scores raised from 19.9 (beam_size=1) to about 22.0 with the beam sizes from 4 to 9. 

Nevertheless, starting from beam size 4, the BLEU gains were minimal while the decoding time kept rising rapidly. The decoding time rose from 19 seconds when using beam size 1 to 51 seconds using beam size 10.In the future, we would opt for beam_size=4 or beam_size=5 for the good BLEU scores and the decoding time.

The highest BLEU score of 22.1, was obtained with a beam size of 9; however, there was little improvement compared to the previous beam sizes. It seems that a beam size of 4 or 5 works well for decoding.


Here is an example:

| Type | Translation |
|---|---|
| Source (EN) | They admire their work. |
| Gold Reference (IT) | Ammirano il lavoro. |
| BPE_8K (beam size = 1) | Affano il loro lavoro. |
| BPE_8K (beam size = 10) | Si ammirano il loro lavoro. |

This example demonstrates that varying the size of the beam may result in differences in the selection of words as a function of the size of the beam. The model with beam_size=1 produces an incorrect choice for the word "Affano", which does not mean the same thing as the source sentence. With beam_size=10, the model produces the choice of the verb "ammirare", which results in a translation that is closest to the reference translation, but still has the translated sentence being grammatically incorrect. 
These results conform to the values reported by BLEU as it can be establishes that larger beams may improve the quality of the translation, but there are diminishing returns for increasing the beam size regarding the additional processing time required for decoding the translated sentences.




**AI Declaration:** GenAI was used to create the tables in this readme for time efficiency.

