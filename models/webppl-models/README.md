Expected output:

> webppl 14-base.js

Posterior belief of the listener about the judgments of speaker and third-party speaker
In these simulations, Adjective0 (A0) is more subjective than Adjective1 (A1).
The speaker is S1, while S2 is a third-party speaker (or perhaps the listener, depending on the interpretation of the model).
...........
After hearing Utterance:   Adjective1 Adjective0 Noun1
Marginals for relevant dimensions (by adjective, speaker, object)
A0(S1,O1)	1	 (Posterior that the speaker judges A0 to hold. Should be high because this adjective came second.)
A0(S2,O1)	0.5199999999999999	 (Posterior that third-party speaker judges A0 to hold. Low, due to low inter-speaker correlation.)
A1(S1,O1)	0.8228571428571427	 (Posterior that the speaker judges A1 to hold. Somewhat lower, since this adjective came first and was subject to loss.)
A1(S2,O1)	0.72	 (Posterior that the third-party speaker judges A1 to hold)

............
After hearing Utterance:   Adjective0 Adjective1 Noun1
Marginals for relevant dimensions (by adjective, speaker, object)
A0(S1,O1)	0.8270676691729324	 (Somewhat lower since A0 was subject to loss)
A0(S2,O1)	0.42105263157894746
A1(S1,O1)	1
A1(S2,O1)	0.9172932330827068	 (High, due to strong inter-speaker correlation)

Speaker Distribution: Subjective adjective is preferred earlier.
Adjective0 Adjective1 Noun1   0.5793646468082411
Adjective1 Adjective0 Noun1   0.4206353531917588
1

