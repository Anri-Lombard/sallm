# C2 AfriHG Hook/Focus Error Table

Created: 2026-05-21.

Purpose: inspect whether the improved Mamba AfriHG beam5/lp1.2 outputs still
fail because they are empty/too short, because they miss the main entity/event,
or because they select the wrong article focus.

This is a decoder-only diagnostic. It does not change the model architecture.

## Artifacts Used

Mamba C1 official test outputs:

- Xho:
  `outputs/eval/diagnostics/c1_mamba_afrihg_beam_lp12_test_20260521/mamba-afrihg-xho-ckpt656-beam5-lp12-test/mamba-afrihg-xho-ckpt656-beam5-lp12-test/examples.jsonl`
- Zul:
  `outputs/eval/diagnostics/c1_mamba_afrihg_beam_lp12_test_20260521/mamba-afrihg-zul-ckpt892-beam5-lp12-test-r2/mamba-afrihg-zul-ckpt892-beam5-lp12-test/examples.jsonl`

Matched C1b LLaMA beam5/lp1.2 outputs:

- Xho:
  `outputs/eval/diagnostics/c1b_llama_afrihg_beam_lp12_test_20260521/llama-afrihg-xho-beam5-lp12-test/llama-afrihg-xho-beam5-lp12-test/examples.jsonl`
- Zul:
  `outputs/eval/diagnostics/c1b_llama_afrihg_beam_lp12_test_20260521/llama-afrihg-zul-beam5-lp12-test/llama-afrihg-zul-beam5-lp12-test/examples.jsonl`

## Heuristic Error Buckets

The buckets below are crude string/overlap heuristics over reference, output,
and the first part of the article. They are for triage, not final metrics.

Xho, n=`1305`:

| Bucket | Count | Share |
|---|---:|---:|
| off-topic/generic | 485 | 37.2% |
| too short/generic | 416 | 31.9% |
| wrong focus | 123 | 9.4% |
| partial topic, wrong focus | 86 | 6.6% |
| repetition/loop | 81 | 6.2% |
| partial topic, short | 61 | 4.7% |
| reasonable partial | 45 | 3.4% |
| empty | 8 | 0.6% |

Zul, n=`1776`:

| Bucket | Count | Share |
|---|---:|---:|
| off-topic/generic | 971 | 54.7% |
| too short/generic | 265 | 14.9% |
| partial topic, wrong focus | 157 | 8.8% |
| wrong focus | 133 | 7.5% |
| partial topic, short | 134 | 7.5% |
| repetition/loop | 65 | 3.7% |
| reasonable partial | 43 | 2.4% |
| empty | 8 | 0.5% |

Preliminary interpretation:

- Beam5/lp1.2 mostly fixes the worst empty/ultra-short behavior but does not
  solve headline planning.
- The dominant residual failure is focus selection: the model often extracts a
  nearby topical phrase or generic news phrase rather than the headline-worthy
  hook.
- Repetition remains visible but is not the dominant failure mode.
- Zul looks more off-topic/generic than Xho by this crude heuristic.
- This points more toward base/task semantic planning and headline formulation
  than toward a pure decoding-length problem.

## Xho First 20 Sequential Examples

The table below is the original first-pass table with older LLaMA output. Keep
it as historical context only; the next section is the matched C1b comparison.

| # | Article hook | Reference | Mamba C1 output | LLaMA old output | Error type |
|---|---|---|---|---|---|
| 1 | Iqela lesizwe loMzantsi Afrika kwiqakamba libuye lintywiz... | Ingxubakaxaka ngo-AB de Villiers | UMzantsi Afrika ifikile | Imidlalo yeNgilane | wrong focus |
| 2 | Umqeqeshi obambeleyo weBanyana Banyana, uDesiree Ellis, u... | ABanyana baya kwitumente ye-Afrika | UMzantsi Afrika uvuyela kwiWomen’s World Cup | Imidlalo ye-Olimpiki | wrong focus |
| 3 | UMotsoeneng uphehlelele lo mbutho phambi kwabantu abangap... | Hlaudi: Kudala abantu bendifuna | UMotsoeneng uyaphikisa uMotsoeneng | jw UMotsoeneng ukholelwa ukuba umhlaba mawunikwe aban... | off-topic/generic |
| 4 | Obesakuba nguMongameli weli uThabo Mbeki uthi ukukhothama... | Thabo Mbeki: Siziva siziinkedama! | Zwelonke Sigcawu: Zwelonke Sigcawu! | tv UMbeki: ‘Siziva siziinkedama’ | repetition/loop |
| 5 | Iqela laseMpuma Koloni, iChippa United, liza kujamelana n... | ‘Ayisonwabisanga indawo elikuyo iqela’ | Kugqumile iChippa United emdlalweni wanamhlanje | zezimali Iqela laseMpuma Koloni, iChippa United, lifu... | wrong focus |
| 6 | URHUQELWE ezinkundleni umasipala ombaxa iNelson Mandela B... | Usiwa enkundleni umasipala ngumbutho iAfriForum | Amathuba iAccess Management emva kokuba iAccess Manag... | zezimali Ibanjiwe iAccess Management | repetition/loop |
| 7 | AbeNkonzo yesiPolisa eMpuma Koloni baphisa ngemali ekumak... | Isixa semali ngokubulawa kukaNamhla | I-R100 000 000 | chunge Kuphisa ngemali ekumakhulu amabini amawaka eer... | off-topic/generic |
| 8 | IZICWANGCISO zokuphuhlisa uluntu oluphantsi komasipala we... | Izicwangciso zaseAmathole ziza kubekwa elubala | Iinzame yokuphuhlisa umasipala waseAmathole | aphakade UMasipala waseAmathole ufuna iNtetho ngoBume... | off-topic/generic |
| 9 | Urhasatsha womdlali omtsha kwiqela lesizwe loMzantsi Afri... | ULungi Ngidi unqwenela into inye jwiii…. | UMzantsi Afrika | chunge Ngidi: Ndinqwenela ukudlala iqakamba | too short/generic |
| 10 | UMHLA wamashumi amabini ananye kuCanzibe ngumhla wokubhiy... | Iphi iti yaseMagwa? | IBhayi le! | chunge Iti yodidi oluphezulu | too short/generic |
| 11 | Ilizwe loMzantsi Afrika lilizwe elihle nabantu balo. Iliz... | Intlungu yesikolo saseMqukwana | ISAZISO UMzantsi Afrika | chunge Isikolo saseMqukwana | off-topic/generic |
| 12 | Batakataka phezu komlambo womchamo nelindle abantwana bel... | Batakataka phezu komchamo nelindle kwaDwesi | Batyakwe abazali babika eMthatha eBhayi! | jw.org Baza kuyijonga le paki! | off-topic/generic |
| 13 | Angene ebaleka AmaNgesi kwiTop 16 yeFifa Wold Cup emva ko... | iNgilane ingombe iWales iibhatyi zagcwala umoya, iMel... | IWales iqokise iWales emdlalweni weFifa | Imidlalo yeFifa Wold Cup | repetition/loop |
| 14 | Ndimfanisa nentyantyambo yasebusika, Yona ingafiyo noba s... | Umzali Wokwenene | Ubethe Wokuqala | aphakade Umzali wokwenene! | too short/generic |
| 15 | Iqela laseKapa, i-Ajax Cape Town, selihlabele mgama ngama... | UChipeta utyikitywe yi-Ajax | Umqobo kwiChippa | Imidlalo ye-Ajax Cape Town | too short/generic |
| 16 | AmaZulu achonge uRomain Folz njengomqeqeshi wawo omtsha. ... | AmaZulu achonge uRomain Folz ngengomqeqeshi omtsha | UFolz noFoluke AmaZulu emdlalweni wanekhaya neChiefs! | zezimali UFolz uncediswa nguAyanda Dlamini | partial topic, wrong focus |
| 17 | Umqeqeshi weMamelodi Sundowns, uManqoba Mngqithi, uthi iq... | UMngqithi uthi ‘Asikabethwa – kodwa besibuthuntu emdl... | UMzantsi Afrika uvuthuntu emdlalweni weDowns! | OSCAR UMngqithi ukhale ngokungasetyenziswa kakuhle kw... | partial topic, short |
| 18 | Ingqatsini yelanga, ebitshisa kwimimandla emininzi kweli ... | Ibiluyolo lodwa kwiilwandle zephondo | Kubhubhile inkosikazi yaseMonti | zezimali Amapolisa aseBhayi avule iqwakuqwaku | off-topic/generic |
| 19 | UMasithembe Rawe uthi kudala warhalela ukukhe akhwele eqo... | ‘Ndifuna siye kuTsolo nesi isithsaba’ | UMzantsi Afrika usendleleni, utsho uMasipala | zezimali UMasithembe Rawe: ‘Ndifuna ntoni? | off-topic/generic |
| 20 | Ndinesicelo wethu siyasokola apha eBhisho; kwingingqi yam... | Ikliniki enye qha! | *Thumela eyakho incwadi eya kuMhleli ngeSMS okanye ku... | xn Isebe lezeMpilo eMpuma Koloni | wrong focus |

## Xho First 20 With Matched C1b LLaMA Beam5/Lp1.2

| # | Article hook | Reference | Mamba C1 output | LLaMA C1b output | Read |
|---|---|---|---|---|---|
| 1 | Iqela lesizwe loMzantsi Afrika kwiqakamba libuye lintywizisa e... | Ingxubakaxaka ngo-AB de Villiers | UMzantsi Afrika ifikile | xn UDe Villiers ubethe iNgilane ngo’3-1 | Mamba too generic/short; LLaMA has more article content. |
| 2 | Umqeqeshi obambeleyo weBanyana Banyana, uDesiree Ellis, ubheng... | ABanyana baya kwitumente ye-Afrika | UMzantsi Afrika uvuyela kwiWomen’s World Cup | ► Umqeqeshi weBanyana Banyana, uDesiree Ellis, ubhengez... | LLaMA over-generates excerpt-like text; Mamba often too generic/short. |
| 3 | UMotsoeneng uphehlelele lo mbutho phambi kwabantu abangaphaya ... | Hlaudi: Kudala abantu bendifuna | UMotsoeneng uyaphikisa uMotsoeneng | neeMbumba zeMpuma Koloni UMotsoeneng ukholelwa ukuba um... | Mamba too generic/short; LLaMA has more article content. |
| 4 | Obesakuba nguMongameli weli uThabo Mbeki uthi ukukhothama kweK... | Thabo Mbeki: Siziva siziinkedama! | Zwelonke Sigcawu: Zwelonke Sigcawu! | xn UMbeki: ‘Siziva siziinkedama’ | Both imperfect; compare focus/entity preservation. |
| 5 | Iqela laseMpuma Koloni, iChippa United, liza kujamelana nowona... | ‘Ayisonwabisanga indawo elikuyo iqela’ | Kugqumile iChippa United emdlalweni wanamhlanje | neeChippa United ziintshatsheli zaseMpuma Koloni | Both imperfect; compare focus/entity preservation. |
| 6 | URHUQELWE ezinkundleni umasipala ombaxa iNelson Mandela Bay, k... | Usiwa enkundleni umasipala ngumbutho iAfriForum | Amathuba iAccess Management emva kokuba iAccess Management | xn Abarhafi besi sixeko bajongene nemiqathango yeAccess... | Both imperfect; compare focus/entity preservation. |
| 7 | AbeNkonzo yesiPolisa eMpuma Koloni baphisa ngemali ekumakhulu ... | Isixa semali ngokubulawa kukaNamhla | I-R100 000 000 | jw.org Ufunyenwe egwintiwe ekhayeni likaNamhla Mtwa | Mamba too generic/short; LLaMA has more article content. |
| 8 | IZICWANGCISO zokuphuhlisa uluntu oluphantsi komasipala wesithi... | Izicwangciso zaseAmathole ziza kubekwa elubala | Iinzame yokuphuhlisa umasipala waseAmathole | ► INtetho kaSodolophu waseAmathole, uZibuthe Mnqwazi, i... | Both imperfect; compare focus/entity preservation. |
| 9 | Urhasatsha womdlali omtsha kwiqela lesizwe loMzantsi Afrika, u... | ULungi Ngidi unqwenela into inye jwiii…. | UMzantsi Afrika | bangalore, india UNgidi uyincomile indlela adlale ngayo... | Mamba too generic/short; LLaMA has more article content. |
| 10 | UMHLA wamashumi amabini ananye kuCanzibe ngumhla wokubhiyozela... | Iphi iti yaseMagwa? | IBhayi le! | neevenkile ezithengisela abantu baseMpuma Koloni | Mamba too generic/short; LLaMA has more article content. |
| 11 | Ilizwe loMzantsi Afrika lilizwe elihle nabantu balo. Ilizwe lo... | Intlungu yesikolo saseMqukwana | ISAZISO UMzantsi Afrika | kuthethwa ngoonqevu! Isikolo saseMqukwana sinabafundi b... | Mamba too generic/short; LLaMA has more article content. |
| 12 | Batakataka phezu komlambo womchamo nelindle abantwana belokish... | Batakataka phezu komchamo nelindle kwaDwesi | Batyakwe abazali babika eMthatha eBhayi! | jw.org Ikomkhulu laseNelson Mandela Bay | Both imperfect; compare focus/entity preservation. |
| 13 | Angene ebaleka AmaNgesi kwiTop 16 yeFifa Wold Cup emva kokubhu... | iNgilane ingombe iWales iibhatyi zagcwala umoya, iMelik... | IWales iqokise iWales emdlalweni weFifa | OSCAR: IWales ne-Iran ziintshatsheli zeFifa World Cup! | Both imperfect; compare focus/entity preservation. |
| 14 | Ndimfanisa nentyantyambo yasebusika, Yona ingafiyo noba seyifi... | Umzali Wokwenene | Ubethe Wokuqala | bangalore: <img border="0" alt="" title="" alt="" src="... | Mamba too generic/short; LLaMA has more article content. |
| 15 | Iqela laseKapa, i-Ajax Cape Town, selihlabele mgama ngamalungi... | UChipeta utyikitywe yi-Ajax | Umqobo kwiChippa | ► Chipeta: Ndibulela uMdali ngokundinika ithuba lokuzib... | LLaMA over-generates excerpt-like text; Mamba often too generic/short. |
| 16 | AmaZulu achonge uRomain Folz njengomqeqeshi wawo omtsha. UFolz... | AmaZulu achonge uRomain Folz ngengomqeqeshi omtsha | UFolz noFoluke AmaZulu emdlalweni wanekhaya neChiefs! | ► Umqeqeshi weMarumo Gallants, u-Ayanda Dlamini, ufuna ... | Both imperfect; compare focus/entity preservation. |
| 17 | Umqeqeshi weMamelodi Sundowns, uManqoba Mngqithi, uthi iqela l... | UMngqithi uthi ‘Asikabethwa – kodwa besibuthuntu emdlal... | UMzantsi Afrika uvuthuntu emdlalweni weDowns! | ► This thesis examines the role of the South African Fo... | LLaMA over-generates excerpt-like text; Mamba often too generic/short. |
| 18 | Ingqatsini yelanga, ebitshisa kwimimandla emininzi kweli phond... | Ibiluyolo lodwa kwiilwandle zephondo | Kubhubhile inkosikazi yaseMonti | © 2018 All Rights Reserved. Designed and Maintained by ... | Mamba too generic/short; LLaMA has more article content. |
| 19 | UMasithembe Rawe uthi kudala warhalela ukukhe akhwele eqongeni... | ‘Ndifuna siye kuTsolo nesi isithsaba’ | UMzantsi Afrika usendleleni, utsho uMasipala | hlanje: UMasithembe Rawe UMasithembe Rawe: ‘Ndiza kubhe... | Mamba generic/wrong focus; LLaMA still imperfect. |
| 20 | Ndinesicelo wethu siyasokola apha eBhisho; kwingingqi yam ikli... | Ikliniki enye qha! | *Thumela eyakho incwadi eya kuMhleli ngeSMS okanye kuWh... | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^... | LLaMA over-generates excerpt-like text; Mamba often too generic/short. |

Matched Xho interpretation:

- Mamba C1 is usually short and often generic or wrong-focus.
- LLaMA C1b often captures more article content and named entities, but the
  same beam5/lp1.2 setting makes it over-generate excerpt-like or noisy text.
- This matches the metric/length result: LLaMA C1b Xho chrF `13.1149` with mean
  prediction length `26.19`, while Mamba C1 Xho chrF is `15.1702` with mean
  prediction length `3.49`.
- Therefore beam5/lp1.2 currently looks like a Mamba-specific AfriHG decoding
  rescue on Xho, not a shared final LLaMA/Mamba protocol.

## Zul First 20 Sequential Examples

The table below is the original first-pass table with older LLaMA output. Keep
it as historical context only; the next section is the matched C1b comparison.

| # | Article hook | Reference | Mamba C1 output | LLaMA old output | Error type |
|---|---|---|---|---|---|
| 1 | IMIKLOMELO yabasebenza emkhakheni wezethelevishini namafi... | Aseqalile amalungiselelo amaSaftas | I-Safta isazodlala kwaSafta amaSafta | I-SA Film & Television Awards izokwethula amaSaftas | off-topic/generic |
| 2 | UMBUSO unike owayeyimeya yeTheku uNkk Zandile Gumede naso... | Liya enkantolo enkulu elowayeyimeya yeTheku | Owe-ANC umbango wezinsizwa iMeya | BUKA: BUKA: BUKA: BUKA: BUKA: BUKA: BUKA... | off-topic/generic |
| 3 | UKHALA ngokuba seholidini kwabadali bakhe umqeqeshi weSup... | UTinkler ukhala ngokuphela komdlandla kubadlali bakhe | UTinkulu kwiPirates lokho uTinkleria Tinkler | U-Eric Tinkler ukhale ngabadlale | off-topic/generic |
| 4 | Damien Chazelle(wenze i Lala Land). RyanGosling (Neil Ams... | I First Man ngeyabathanda ukwazi ngomhlaba | [IMIBONO NEWENDIHLWENI-First Man - FirstMan | chunge [IMIBONO] Isexwayiso ku-Arsmtrong | partial topic, wrong focus |
| 5 | UZINIKELE emaphoyiseni umlisa (28) waseMgungundlovu osolw... | Uzinikele emaphoyiseni ‘onqume’ ingane | Usolwa ngokugathi’ uzakwabo’ | jw.org Usolwa ngokugxambukela ezindabeni zakhe | off-topic/generic |
| 6 | ZAKHELE XABAUZITHWESE ijoka lokuwubhala kabusha umlando n... | UMwekassa ufuna isicoco | uqophe uchazwa kowesibhakela | kmen UZAKHELE XABAUZITHWESE ijoka lokuwubhala kabusha... | off-topic/generic |
| 7 | NGQESHE BUTHELEZIE MINYAKENI eminingi ngibhala ngezimoto,... | AbakwaMazda benze okungajwayelekile | I-Mazda 3Mazda 2-Mazda | chunge [IMIBONO] I-Mazda ihlukene ngezinhlobo eziyisi... | off-topic/generic |
| 8 | LISHONE elayizolo 'omakoti' okungamaqembu epolitiki amanc... | #Ukhetho2019: Okushiwo abahlaziyi 'ngomakoti' kwipoli... | Kusele kowe-ANC ukulwela kwi-ANC | I-DA ne-ACDP kuvalwe umlomo | off-topic/generic |
| 9 | INSIZWA esizakhele igama kuleli ngokuba usomabhizinisi on... | U-DJ Sbu usekulungele ukushadelwa | U-Sbuya Dreams ethusayo’sibano | Usomabhizinisi onegama elinguDJ Sbu | off-topic/generic |
| 10 | UMSAKAZI weGagasi FM, uFelix Hlophe, usekhwele wadilika k... | UFelix usola uKini ngokudicilela phansi isithunzi sak... | UFelix Hlophe usolwa kuKini | OweGagasi FM usekhwele wadilika kuKini Shandu | partial topic, short |
| 11 | Umcimbi wakulo nyaka uzoba ngoJulayi 6, ePeople’s Park eM... | Udinga uxhaso uDJ Tira ngeFact Durban Rocks | UTira ozoMzansi | Umcimbi uzoba ngoJulayi 6 | too short/generic |
| 12 | Ngifisa sengathi nalapha eNingizimu Afrika singabona isix... | Abathenwe abantu abadlwengulayo | ] LOTHANDO L] Likhulu') Liningi' ubulungiswa' emantfu... | [IMIBONO] Ngifisa sengathi nalapha eNingizimu Afrika ... | off-topic/generic |
| 13 | UBALISA ngobunzima bokuhlowa nsuku zonke ngenxa yeCovid-1... | 'Kunyomfeka izinhlelo ngenxa yeCovid-19' | UMaduka igebe kwi-United: Maduka | UMaduka uqeqesha iBloemfontein Celtic | off-topic/generic |
| 14 | UMQEQESHI we-Ajax Cape Town, uStanely Menzo, uthe iqembu ... | Ubona ihlazo ngokushaywa yiKwaDukuza | I-Ajax abeKwaDukuza kwi-Ajax | Lwezandla I-Ajax izogxila ku-Ajax | off-topic/generic |
| 15 | ZAKHELE XABAUMDLALI wasesiswini weCape Town City, uSurpri... | ‘Bamfikisa’ uSurprise Ralani eqala ukufika kwiCape To... | URalani uvule eyeligi: Ralani | USurprise Ralani uvule isifuba | partial topic, short |
| 16 | AKAZIBONI eyeka ukudlala iLotto owesilisa waseMitchells P... | Akaziboni eyeka ukudlala iLotto oseyibambe kabili kul... | U-R75 000 obekwiwayini u-R75 000 | U-R75 000 uzoshintsha impilo yomndeni wakhe | repetition/loop |
| 17 | UNtando, owabopha ifindo likasofa silahlane nomculi uKhay... | UmkaKhaya Mthethwa usafisa ukufeza amaphupho | UNtando uzishaya isifuba ngoMiss SA | chunge UNtando uboshelwe ukushada | off-topic/generic |
| 18 | IZINYUNYANA zothisha zisho ngazwi linye ukuthi amalungu a... | Baxwayisa ngokuvulwa kwezikole | Umnyango zibuyela emsebenzini ngenxa yeCorona | Izinyunyana zifuna uhulumeni agcine izethembiso zakhe | wrong focus |
| 19 | KUSOLWA ukuthi uyiphoyisa owesilisa ongene eNkantolo yesi... | Iphoyisa lidubule labulala ababili enkantolo kwelediv... | Uvele enkantolo kwelokishini | jw.org Kuboshwe owesilisa ongene eNkantolo yesifunda | partial topic, short |
| 20 | INGODUSO yowayengumengameli wakuleli, uNksz Nonkanyiso Co... | Ingoduso kaZuma ikhala ngomzwangedwa | UZuma ebhokile' ekhaya uZuma | U-Conco uchitha eyakhe impilo | off-topic/generic |

## Zul First 20 With Matched C1b LLaMA Beam5/Lp1.2

| # | Article hook | Reference | Mamba C1 output | LLaMA C1b output | Read |
|---|---|---|---|---|---|
| 1 | IMIKLOMELO yabasebenza emkhakheni wezethelevishini namafilimu amaSout... | Aseqalile amalungiselelo amaSaftas | I-Safta isazodlala kwaSafta amaSafta | sexuality I-SA Film and Television Awards izokwethula amaSaftas akulo... | Both imperfect; compare focus/entity preservation. |
| 2 | UMBUSO unike owayeyimeya yeTheku uNkk Zandile Gumede nasolwa nabo ama... | Liya enkantolo enkulu elowayeyimeya yeTheku | Owe-ANC umbango wezinsizwa iMeya | BUKA: BUKA: BUKA: BUKA: BUKA: BUKA: BUKA: BUKA: BUKA: BUKA: BUKA: BUK... | LLaMA over-generates; Mamba shorter/generic. |
| 3 | UKHALA ngokuba seholidini kwabadali bakhe umqeqeshi weSuperSport Unit... | UTinkler ukhala ngokuphela komdlandla kubadlali bakhe | UTinkulu kwiPirates lokho uTinkleria Tinkler | U-Eric Tinkler ukhale ngabadali bakhe | Both imperfect; compare focus/entity preservation. |
| 4 | Damien Chazelle(wenze i Lala Land). RyanGosling (Neil Amstrong); Clai... | I First Man ngeyabathanda ukwazi ngomhlaba | [IMIBONO NEWENDIHLWENI-First Man - FirstMan | [IMIBONO] UNeil Amstrong uchaza ngomlando ka-Arsmtrong [IMIBONO] U-Ar... | LLaMA over-generates; Mamba shorter/generic. |
| 5 | UZINIKELE emaphoyiseni umlisa (28) waseMgungundlovu osolwa ngokuthumb... | Uzinikele emaphoyiseni ‘onqume’ ingane | Usolwa ngokugathi’ uzakwabo’ | sexual abuse Usolwa ngokugxambukela ezindabeni zakhe | Both imperfect; compare focus/entity preservation. |
| 6 | ZAKHELE XABAUZITHWESE ijoka lokuwubhala kabusha umlando nge-Afrika um... | UMwekassa ufuna isicoco | uqophe uchazwa kowesibhakela | UZAKHELE XABAUZITHWESE ijoka lokuwubhala kabusha umlando nge-Afrika | Mamba too short/generic; LLaMA has more content. |
| 7 | NGQESHE BUTHELEZIE MINYAKENI eminingi ngibhala ngezimoto, bekuqala uk... | AbakwaMazda benze okungajwayelekile | I-Mazda 3Mazda 2-Mazda | [IMIBONO] I-Mazda ihlukene ngezinhlobo eziyisithupha zakwaMazda I-Maz... | Both imperfect; compare focus/entity preservation. |
| 8 | LISHONE elayizolo 'omakoti' okungamaqembu epolitiki amancane, amanye ... | #Ukhetho2019: Okushiwo abahlaziyi 'ngomakoti' kwipolitiki | Kusele kowe-ANC ukulwela kwi-ANC | I-EFF ne-Azanian People’s Organisation I-EFF ne-Azanian People’s Orga... | LLaMA over-generates; Mamba shorter/generic. |
| 9 | INSIZWA esizakhele igama kuleli ngokuba usomabhizinisi onegama, abani... | U-DJ Sbu usekulungele ukushadelwa | U-Sbuya Dreams ethusayo’sibano | Usomabhizinisi onegama engimthanda ngayo yonke inhliziyo | Both imperfect; compare focus/entity preservation. |
| 10 | UMSAKAZI weGagasi FM, uFelix Hlophe, usekhwele wadilika kuKini Shandu... | UFelix usola uKini ngokudicilela phansi isithunzi sakhe | UFelix Hlophe usolwa kuKini | OweGagasi FM usekhwele wadilika kuKini Shandu | Mamba too short/generic; LLaMA has more content. |
| 11 | Umcimbi wakulo nyaka uzoba ngoJulayi 6, ePeople’s Park eMoses Mabhida... | Udinga uxhaso uDJ Tira ngeFact Durban Rocks | UTira ozoMzansi | Umcimbi uzoba ngoJulayi 6 | Mamba too short/generic; LLaMA has more content. |
| 12 | Ngifisa sengathi nalapha eNingizimu Afrika singabona isixazululo esiz... | Abathenwe abantu abadlwengulayo | ] LOTHANDO L] Likhulu') Liningi' ubulungiswa' emantfuka' emantwela'). | [IMIBONO] Ngifisa sengathi nalapha eNingizimu Afrika singabona isixaz... | LLaMA over-generates; Mamba shorter/generic. |
| 13 | UBALISA ngobunzima bokuhlowa nsuku zonke ngenxa yeCovid-19, uJohn Mad... | 'Kunyomfeka izinhlelo ngenxa yeCovid-19' | UMaduka igebe kwi-United: Maduka | UMaduka uchitha eyokuhlolelwa iCovid-19 | Both imperfect; compare focus/entity preservation. |
| 14 | UMQEQESHI we-Ajax Cape Town, uStanely Menzo, uthe iqembu lakhe lizifa... | Ubona ihlazo ngokushaywa yiKwaDukuza | I-Ajax abeKwaDukuza kwi-Ajax | I-Ajax izogxila ku-Ajax Cape Town | Both imperfect; compare focus/entity preservation. |
| 15 | ZAKHELE XABAUMDLALI wasesiswini weCape Town City, uSurprise Ralani, u... | ‘Bamfikisa’ uSurprise Ralani eqala ukufika kwiCape Town City | URalani uvule eyeligi: Ralani | kutakula uSurprise Ralani | Both imperfect; compare focus/entity preservation. |
| 16 | AKAZIBONI eyeka ukudlala iLotto owesilisa waseMitchells Plain eKapa o... | Akaziboni eyeka ukudlala iLotto oseyibambe kabili kuleli | U-R75 000 obekwiwayini u-R75 000 | U-R75 000 uzoshintsha impilo yomndeni wakhe | Both imperfect; compare focus/entity preservation. |
| 17 | UNtando, owabopha ifindo likasofa silahlane nomculi uKhaya Mthethwa n... | UmkaKhaya Mthethwa usafisa ukufeza amaphupho | UNtando uzishaya isifuba ngoMiss SA | UNtando ubopha ifindo likasofa silahlane nomyeni wakhe | Both imperfect; compare focus/entity preservation. |
| 18 | IZINYUNYANA zothisha zisho ngazwi linye ukuthi amalungu azobuyela ems... | Baxwayisa ngokuvulwa kwezikole | Umnyango zibuyela emsebenzini ngenxa yeCorona | Izinyunyana zifuna uhulumeni agcine izethembiso zakhe zokuvikela abas... | Both imperfect; compare focus/entity preservation. |
| 19 | KUSOLWA ukuthi uyiphoyisa owesilisa ongene eNkantolo yesifunda eThekw... | Iphoyisa lidubule labulala ababili enkantolo kweledivosi | Uvele enkantolo kwelokishini | tv Kuboshwe owesilisa ongene eNkantolo yesifunda eThekwini | Mamba too short/generic; LLaMA has more content. |
| 20 | INGODUSO yowayengumengameli wakuleli, uNksz Nonkanyiso Conco, yethuse... | Ingoduso kaZuma ikhala ngomzwangedwa | UZuma ebhokile' ekhaya uZuma | U-Conco usola uZuma ngokufa kukaNksz Conco | Both imperfect; compare focus/entity preservation. |

Matched Zul interpretation:

- Mamba C1 is concise and close to headline length, but often generic,
  malformed, repetitive around a named entity, or wrong-focus.
- LLaMA C1b often preserves more surface article content than Mamba, but
  beam5/lp1.2 makes it too long for the headline task and sometimes noisy.
- Metric/length summary using whitespace tokens: LLaMA C1b Zul chrF `20.8709`
  with mean prediction length `11.74` versus reference `4.76`; Mamba C1 Zul
  chrF `17.1245` with mean prediction length `3.60` versus reference `4.76`.
- LLaMA C1b Zul is still below the tracked current-stack LLaMA Zul parity
  value `21.5552` and older best `23.0041`, so this decode setting is not a
  shared LLaMA improvement even though LLaMA remains stronger than Mamba on Zul.

## Decision

- The improved Mamba AfriHG recipe still fails mainly through focus selection,
  generic headline selection, and weak entity/event planning.
- It is not just a formatting or empty-generation problem after beam5/lp1.2.
- Decoding helped substantially, but the remaining gap likely needs better base
  quality and/or a better decoder-only headline formulation, not only more
  length control.
- Matched LLaMA examples show that beam5/lp1.2 is not a shared AfriHG decode
  protocol: it over-generates or becomes noisy for LLaMA, while helping Mamba
  move closer to headline length.
- Carry-forward tag: `Mamba-only` for the current AfriHG beam5/lp1.2 rescue;
  still requires explicit labeling in final fair Mamba-vs-LLaMA comparison.

Optional reporting follow-up:

- Add a smaller advisor-facing table with the most illustrative cases, rather
  than only the first 20 sequential examples.
