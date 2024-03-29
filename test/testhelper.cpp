//
// Created by breucking on 27.12.21.
//


#include "testhelper.h"

using json = nlohmann::json;

/*
 * Define some test data
 */
std::vector<std::string> DICT = {"root-asset", "date", "null", "type", "blog", "author", "washington post staff",
                                 "source", "the washington post", "title", "benghazi", "libya", "transcript", "skip",
                                 "quorum", "alert", "witness", "deserve", "decorum", "disrupt", "proper", "incremental",
                                 "ranking", "recognized", "stevens", "wood", "courage", "gate", "machine", "mortar",
                                 "pour", "sacrifice", "everlasting", "gratitude", "owe", "anniversary", "9/11",
                                 "compound", "pursuing", "narrow", "scope", "linger", "incapable", "unwilling",
                                 "relevant", "accounting", "mail", "knowledgeable", "server", "custody", "uncover",
                                 "testimony", "particularity", "accountability", "thorough", "ambassador", "frank",
                                 "soil", "fundamental", "obligation", "purport", "hundred", "arrangement", "exclusive",
                                 "sole", "amendment", "privilege", "incrimination", "annals", "experiment", "examine",
                                 "pledged", "manner", "worthy", "memory", "pursue", "definitive", "profile", "brave",
                                 "bipartisan", "respected", "caucus", "deadline", "unlimited", "squandering",
                                 "taxpayer", "derail", "reasonable", "abusive", "credibility", "mccarthy", "speaker",
                                 "zero", "cross", "quote", "justified", "jurisdictional", "crossed", "dying", "graham",
                                 "hell", "wild", "objection", "database", "forth", "spin", "conspiracy", "accepting",
                                 "inaccurate", "sidney", "checker", "rating", "deposit", "fishing", "expedition",
                                 "corps", "diplomatic", "yield", "gentleman", "admired", "diplomat", "accomplished",
                                 "sand", "shoe", "liked", "revolution", "envoy", "gathering", "libyan", "dictator",
                                 "greek", "cargo", "ship", "diplomacy", "testament", "awareness", "ten", "islam",
                                 "prophet", "personally", "valued", "pretoria", "baghdad", "montreal", "outpost",
                                 "distance", "afghanistan", "trained", "distinguished", "globe", "haiti", "tenure",
                                 "painful", "marines", "casket", "plane", "andrews", "tragedy", "prior", "hard-hitting",
                                 "boot", "operate", "understood", "terrorism", "achieve", "inevitably", "contractor",
                                 "locally", "employed", "bunker", "devoted", "root", "aggressor", "vacuum",
                                 "everywhere", "egypt", "saudi", "jerusalem", "intifada", "extremist", "instability",
                                 "rampant", "pivotal", "unrest", "democracy", "eastern", "fragile", "foothold",
                                 "tunisia", "destabilize", "airplane", "sky", "israeli", "airport", "nearest",
                                 "dignity", "anywhere", "substitute", "embassy", "resort", "creative", "profession",
                                 "dedicated", "observation", "safely", "branch", "deadly", "reagan", "marine",
                                 "barrack", "bomb", "kenya", "tanzania", "longest", "appointed", "institution",
                                 "boards", "findings", "africa", "punch", "implementation", "slate", "deployment",
                                 "deploy", "subsequent", "joint", "pride", "ideology", "abroad", "cooperation",
                                 "landmark", "treaty", "myanmar", "democratic", "patriotism", "loyalty", "resist",
                                 "disagree", "partisan", "bestow", "statesmanship", "shrift", "precis", "appreciated",
                                 "illinois", "sullivan", "memo", "tripoli", "rebel", "tightening", "noose",
                                 "instrumental", "pause", "considerable", "boil", "arab", "rose", "chart", "demanding",
                                 "genocide", "hunt", "cockroach", "intensity", "shore", "strongly", "idly",
                                 "unintended", "diligence", "heading", "enormous", "dig", "casualty", "remarkable",
                                 "intervention", "mull", "tactical", "courtesy", "obstacle", "gates", "advocate",
                                 "urging", "in-person", "assumed", "resolution", "unprecedented", "instruction",
                                 "capacity", "bulk", "militarily", "convinced", "veto", "persuade", "abstain",
                                 "congressman", "joining", "legitimacy", "behest", "varying", "bin", "intense",
                                 "inaudible", "humanitarian", "launching", "strategic", "totality", "overcome", "drove",
                                 "load", "vital", "recall", "monetary", "sum", "interchange", "architect", "disaster",
                                 "yesterday", "unanswered", "implication", "rightly", "handled", "nonpartisan",
                                 "importantly", "judgment", "conducting", "lamb", "supervisor", "cable", "signature",
                                 "falsely", "clip", "outright", "whopper", "copy", "stamp", "compile", "charleston",
                                 "detailed", "confusion", "clarify", "tradition", "actionable", "accident",
                                 "historically", "scrutiny", "indiana", "ms", "brooks", "drawing", "pile", "relate",
                                 "hourly", "handful", "disparity", "seated", "conclude", "representative", "hand-pick",
                                 "paris", "reconnaissance", "insurgency", "fashioned", "anticipated", "assessment",
                                 "malta", "groundwork", "lay", "recollection", "refresh", "interrupt", "expeditionary",
                                 "undertaking", "practiced", "communication", "instantaneous", "undertake", "anxious",
                                 "depart", "worsened", "guide", "overtly", "constant", "radar", "explosive", "device",
                                 "behalf", "classified", "bet", "impression", "sharing", "thank", "informed",
                                 "briefing", "one-on-one", "weekly", "briefcase", "occasions", "recommendation",
                                 "conjunction", "questioning", "procedure", "handling", "dedication", "overseas",
                                 "affinity", "handshake", "globally", "safeguard", "wherever", "heroic", "ideal",
                                 "sufficient", "contingency", "communications", "assets", "assess", "dempsey",
                                 "guidance", "kabul", "incredibly", "supportive", "snapshot", "counterpart", "carter",
                                 "useful", "institutionalized", "platoon", "periodic", "waste", "latter", "impress",
                                 "institute", "reward", "litany", "fence", "lady", "gentle", "alabama", "uncertainly",
                                 "uncertainty", "colleague", "belong", "noticed", "desk", "tab", "encounter",
                                 "lockdown", "mrs", "stacks", "chemical", "rid", "proliferation", "collected",
                                 "destroyed", "reducing", "stocks", "burns", "please", "visiting", "numerous",
                                 "consulate", "vigorous", "formal", "drill", "brush", "frustrating", "correspond",
                                 "assigned", "bass", "acknowledge", "civic", "permission", "requested", "repel",
                                 "unfortunate", "annex", "obsession", "formed", "hat", "shame", "suicide",
                                 "embarrassing", "pakistan", "yemen", "willingness", "traveled", "amazed", "chafe",
                                 "peshawar", "vividly", "driven", "bias", "ear", "blind", "guarded", "journal",
                                 "function", "editorial", "orderly", "dysfunction", "substance", "stalk", "arkansas",
                                 "interpreter", "bunch", "militant", "protected", "referenced", "spoken", "publication",
                                 "stability", "equipped", "mine", "practically", "rotated", "investigate", "supplement",
                                 "fortification", "unfortunately", "kansas", "summary", "paycheck", "constantly",
                                 "connected", "breach", "duty", "legally", "absence", "dereliction", "channels",
                                 "occasionally", "slope", "perspective", "poster", "fulfilled", "namely", "flag",
                                 "allegedly", "wittingly", "unwittingly", "sanchez", "allegation", "volume", "emphasis",
                                 "hysterical", "occasion", "observe", "jacob", "queue", "mitchell", "logic",
                                 "falsehood", "correctly", "description", "factual", "factually", "incorrect", "theory",
                                 "aftermath", "myth", "debunk", "rand", "documentation", "counter", "shortly",
                                 "defending", "whirl", "evacuate", "distressing", "congress", "indicate", "directed",
                                 "deck", "nowhere", "cairo", "victoria", "spokesperson", "spray", "demonstration",
                                 "eyewitness", "rice", "consequence", "spontaneous", "false", "reservation",
                                 "narrative", "roof", "grave", "egyptian", "inflammatory", "attacker", "justification",
                                 "tunis", "thankfully", "appreciate", "pressed", "approximately", "fluid", "affiliated",
                                 "conflicting", "slide", "rhodes", "assume", "alive", "square", "chapter", "disservice",
                                 "glad", "confusing", "motivated", "khartoum", "burning", "tunisian", "desperate",
                                 "insinuation", "plain", "vastly", "monitor", "sake", "scheme", "entitled", "premium",
                                 "documentary", "buck", "exhibit", "ceiling", "recitation", "sheer", "metric",
                                 "regularity", "amazing", "centric", "aired", "cancel", "conclusion", "investigator",
                                 "ignore", "damn", "singly", "briefly", "merit", "confused", "accusation", "sleep",
                                 "wrack", "brain", "predecessor", "nairobi", "alike", "east", "optimist", "prosecution",
                                 "stack", "sad", "regardless", "undisputed", "bubble", "solicit", "redact",
                                 "identifier", "insight", "keeper", "tomorrow", "intervene", "electricity",
                                 "definition", "evaluate", "print", "determination", "somewhere", "thread", "prolific",
                                 "explanation", "tyler", "rhetoric", "derision", "seizing", "jaw", "unenthusiastic",
                                 "paralysis", "lukewarm", "operational", "pseudo", "venture", "madam", "drivel",
                                 "eminently", "relevance", "classification", "subpoena", "deposition", "compelling",
                                 "unilaterally", "marshal", "sir", "morse", "carrier", "pigeon", "smoke", "irrelevant",
                                 "telephone", "bothered", "motion", "parliamentarian", "mills", "fairness", "discovery",
                                 "odd", "outstanding", "admission", "selectively", "indulgence", "signify", "clerk",
                                 "adequately", "soft", "mailing", "react", "favorable", "avenue", "best", "newly",
                                 "lever", "register", "follower", "incoming", "adamantly", "convey", "engage",
                                 "messaging", "folder", "medicine", "gasoline", "diesel", "milk", "visa", "basically",
                                 "affirmatively", "positively", "fuel", "insult", "servant", "similarly", "elements",
                                 "unofficial", "originally", "buttonhole", "reception", "helpful", "carefully",
                                 "incredulity", "testify", "aisle", "overwhelm", "somehow", "posed", "historical",
                                 "influential", "deference", "consent", "reopen", "full-time", "sentiment", "inhabit",
                                 "permanent", "promote", "vibrant", "bilateral", "outweigh", "elimination",
                                 "profoundly", "lacking", "minimal", "nonexistent", "configuration", "geography",
                                 "mid-july", "formally", "hiding", "headquarters", "accurate", "duration", "dispute",
                                 "predominantly", "located", "informative", "update", "gradual", "deem", "consul",
                                 "protocol", "accurately", "disclosed", "assignment", "militia", "disarm", "miss",
                                 "dated", "universe", "curious", "humor", "typing", "picking", "spirit",
                                 "entrepreneurial", "barrier", "length", "describe", "patrick", "high-level",
                                 "evaluation", "advised", "abandon", "contrary", "unaware", "prioritize", "farthest",
                                 "unquestionably", "viewer", "embarrassment", "reference", "prosecute", "propel",
                                 "passionate", "rip", "remove", "juxtapose", "difficulty", "drag", "atmosphere",
                                 "depth", "specify", "undergo", "circle", "aunt", "remainder", "haven", "inflict",
                                 "inhalation", "enable", "endeavor", "interior", "proceeding", "instinct",
                                 "desperately", "saving", "succumb", "horror", "resuscitate", "labored", "fog",
                                 "heroism", "professionalism", "refuge", "fortified", "whisper", "fighter", "god",
                                 "one-tenth", "row", "forbid", "soliloquy", "elegant", "finger", "tape", "rewind",
                                 "regret", "violation", "cell", "24/7", "accordance", "respects", "breakfast", "trick",
                                 "follow-up", "inadequate", "grossly", "magnitude", "preventable", "premises",
                                 "vulnerable", "lease", "consensus", "west", "professionalize", "height", "province",
                                 "requirement", "dip", "dialogue", "boring", "pedigree", "honorable", "malign", "sic",
                                 "retreated", "accountable", "oversight", "mismanagement", "price", "inflexible",
                                 "limb", "armor", "static", "kinetic", "flexibility", "vet", "reliable", "invite",
                                 "friendly", "accompany", "landed", "firearm", "vienna", "trigger", "funded",
                                 "responsiveness", "echo", "expression", "havoc", "causing", "therefore", "explicit",
                                 "berlin", "rome", "istanbul", "appoint", "laying", "foundation", "timeline", "bother",
                                 "key", "explicitly", "comprehensive", "excerpt", "cabinet", "instruct", "recognition",
                                 "vacation", "handed", "guarantee", "driveway", "timing", "suggestion", "rescue",
                                 "import", "insofar", "doctrine", "publicize", "coalition", "sanction", "imposed",
                                 "disposal", "extent", "lap", "precise", "recess", "mess", "bolster", "approve",
                                 "approving", "premise", "distinction", "listening", "chaos", "spike", "cross2",
                                 "forthcoming", "monitoring", "consideration", "departure", "ultimate", "oversee",
                                 "properly", "fault", "mislead", "pleased", "beef", "maximum", "informing", "tourist",
                                 "criterion", "extreme", "conditioning", "orientation", "imperative", "gavel",
                                 "quicker", "attached", "run-up", "append", "hampshire", "asset", "spinning",
                                 "suspicion", "leon", "undeniable", "intentionally", "pale", "surveillance", "drone",
                                 "ordering", "rota", "spain", "croatia", "mobilize", "redirect", "bound", "dispatch",
                                 "satisfied", "dissipated", "actively", "assertion", "scrambled", "logistics",
                                 "upheaval", "volatility", "amateur", "mock", "afghan", "tire", "indonesian", "jakarta",
                                 "volatile", "prayer", "battering", "army", "fancy"};
std::vector<std::string> DICT2 = {"root-asset1", "date1", "null1", "type1", "blog1", "author1",
                                  "washington post staff1", "source1", "the washington post1", "title1", "benghazi1",
                                  "libya1", "transcript1", "skip1", "quorum1", "alert1", "witness1", "deserve1",
                                  "decorum1", "disrupt1", "proper1", "incremental1", "ranking1", "recognized1",
                                  "stevens1", "wood1", "courage1", "gate1", "machine1", "mortar1", "pour1",
                                  "sacrifice1", "everlasting1", "gratitude1", "owe1", "anniversary1", "9/111",
                                  "compound1", "pursuing1", "narrow1", "scope1", "linger1", "incapable1", "unwilling1",
                                  "relevant1", "accounting1", "mail1", "knowledgeable1", "server1", "custody1",
                                  "uncover1", "testimony1", "particularity1", "accountability1", "thorough1",
                                  "ambassador1", "frank1", "soil1", "fundamental1", "obligation1", "purport1",
                                  "hundred1", "arrangement1", "exclusive1", "sole1", "amendment1", "privilege1",
                                  "incrimination1", "annals1", "experiment1", "examine1", "pledged1", "manner1",
                                  "worthy1", "memory1", "pursue1", "definitive1", "profile1", "brave1", "bipartisan1",
                                  "respected1", "caucus1", "deadline1", "unlimited1", "squandering1", "taxpayer1",
                                  "derail1", "reasonable1", "abusive1", "credibility1", "mccarthy1", "speaker1",
                                  "zero1", "cross1", "quote1", "justified1", "jurisdictional1", "crossed1", "dying1",
                                  "graham1", "hell1", "wild1", "objection1", "database1", "forth1", "spin1",
                                  "conspiracy1", "accepting1", "inaccurate1", "sidney1", "checker1", "rating1",
                                  "deposit1", "fishing1", "expedition1", "corps1", "diplomatic1", "yield1",
                                  "gentleman1", "admired1", "diplomat1", "accomplished1", "sand1", "shoe1", "liked1",
                                  "revolution1", "envoy1", "gathering1", "libyan1", "dictator1", "greek1", "cargo1",
                                  "ship1", "diplomacy1", "testament1", "awareness1", "ten1", "islam1", "prophet1",
                                  "personally1", "valued1", "pretoria1", "baghdad1", "montreal1", "outpost1",
                                  "distance1", "afghanistan1", "trained1", "distinguished1", "globe1", "haiti1",
                                  "tenure1", "painful1", "marines1", "casket1", "plane1", "andrews1", "tragedy1",
                                  "prior1", "hard-hitting1", "boot1", "operate1", "understood1", "terrorism1",
                                  "achieve1", "inevitably1", "contractor1", "locally1", "employed1", "bunker1",
                                  "devoted1", "root1", "aggressor1", "vacuum1", "everywhere1", "egypt1", "saudi1",
                                  "jerusalem1", "intifada1", "extremist1", "instability1", "rampant1", "pivotal1",
                                  "unrest1", "democracy1", "eastern1", "fragile1", "foothold1", "tunisia1",
                                  "destabilize1", "airplane1", "sky1", "israeli1", "airport1", "nearest1", "dignity1",
                                  "anywhere1", "substitute1", "embassy1", "resort1", "creative1", "profession1",
                                  "dedicated1", "observation1", "safely1", "branch1", "deadly1", "reagan1", "marine1",
                                  "barrack1", "bomb1", "kenya1", "tanzania1", "longest1", "appointed1", "institution1",
                                  "boards1", "findings1", "africa1", "punch1", "implementation1", "slate1",
                                  "deployment1", "deploy1", "subsequent1", "joint1", "pride1", "ideology1", "abroad1",
                                  "cooperation1", "landmark1", "treaty1", "myanmar1", "democratic1", "patriotism1",
                                  "loyalty1", "resist1", "disagree1", "partisan1", "bestow1", "statesmanship1",
                                  "shrift1", "precis1", "appreciated1", "illinois1", "sullivan1", "memo1", "tripoli1",
                                  "rebel1", "tightening1", "noose1", "instrumental1", "pause1", "considerable1",
                                  "boil1", "arab1", "rose1", "chart1", "demanding1", "genocide1", "hunt1", "cockroach1",
                                  "intensity1", "shore1", "strongly1", "idly1", "unintended1", "diligence1", "heading1",
                                  "enormous1", "dig1", "casualty1", "remarkable1", "intervention1", "mull1",
                                  "tactical1", "courtesy1", "obstacle1", "gates1", "advocate1", "urging1", "in-person1",
                                  "assumed1", "resolution1", "unprecedented1", "instruction1", "capacity1", "bulk1",
                                  "militarily1", "convinced1", "veto1", "persuade1", "abstain1", "congressman1",
                                  "joining1", "legitimacy1", "behest1", "varying1", "bin1", "intense1", "inaudible1",
                                  "humanitarian1", "launching1", "strategic1", "totality1", "overcome1", "drove1",
                                  "load1", "vital1", "recall1", "monetary1", "sum1", "interchange1", "architect1",
                                  "disaster1", "yesterday1", "unanswered1", "implication1", "rightly1", "handled1",
                                  "nonpartisan1", "importantly1", "judgment1", "conducting1", "lamb1", "supervisor1",
                                  "cable1", "signature1", "falsely1", "clip1", "outright1", "whopper1", "copy1",
                                  "stamp1", "compile1", "charleston1", "detailed1", "confusion1", "clarify1",
                                  "tradition1", "actionable1", "accident1", "historically1", "scrutiny1", "indiana1",
                                  "ms1", "brooks1", "drawing1", "pile1", "relate1", "hourly1", "handful1", "disparity1",
                                  "seated1", "conclude1", "representative1", "hand-pick1", "paris1", "reconnaissance1",
                                  "insurgency1", "fashioned1", "anticipated1", "assessment1", "malta1", "groundwork1",
                                  "lay1", "recollection1", "refresh1", "interrupt1", "expeditionary1", "undertaking1",
                                  "practiced1", "communication1", "instantaneous1", "undertake1", "anxious1", "depart1",
                                  "worsened1", "guide1", "overtly1", "constant1", "radar1", "explosive1", "device1",
                                  "behalf1", "classified1", "bet1", "impression1", "sharing1", "thank1", "informed1",
                                  "briefing1", "one-on-one1", "weekly1", "briefcase1", "occasions1", "recommendation1",
                                  "conjunction1", "questioning1", "procedure1", "handling1", "dedication1", "overseas1",
                                  "affinity1", "handshake1", "globally1", "safeguard1", "wherever1", "heroic1",
                                  "ideal1", "sufficient1", "contingency1", "communications1", "assets1", "assess1",
                                  "dempsey1", "guidance1", "kabul1", "incredibly1", "supportive1", "snapshot1",
                                  "counterpart1", "carter1", "useful1", "institutionalized1", "platoon1", "periodic1",
                                  "waste1", "latter1", "impress1", "institute1", "reward1", "litany1", "fence1",
                                  "lady1", "gentle1", "alabama1", "uncertainly1", "uncertainty1", "colleague1",
                                  "belong1", "noticed1", "desk1", "tab1", "encounter1", "lockdown1", "mrs1", "stacks1",
                                  "chemical1", "rid1", "proliferation1", "collected1", "destroyed1", "reducing1",
                                  "stocks1", "burns1", "please1", "visiting1", "numerous1", "consulate1", "vigorous1",
                                  "formal1", "drill1", "brush1", "frustrating1", "correspond1", "assigned1", "bass1",
                                  "acknowledge1", "civic1", "permission1", "requested1", "repel1", "unfortunate1",
                                  "annex1", "obsession1", "formed1", "hat1", "shame1", "suicide1", "embarrassing1",
                                  "pakistan1", "yemen1", "willingness1", "traveled1", "amazed1", "chafe1", "peshawar1",
                                  "vividly1", "driven1", "bias1", "ear1", "blind1", "guarded1", "journal1", "function1",
                                  "editorial1", "orderly1", "dysfunction1", "substance1", "stalk1", "arkansas1",
                                  "interpreter1", "bunch1", "militant1", "protected1", "referenced1", "spoken1",
                                  "publication1", "stability1", "equipped1", "mine1", "practically1", "rotated1",
                                  "investigate1", "supplement1", "fortification1", "unfortunately1", "kansas1",
                                  "summary1", "paycheck1", "constantly1", "connected1", "breach1", "duty1", "legally1",
                                  "absence1", "dereliction1", "channels1", "occasionally1", "slope1", "perspective1",
                                  "poster1", "fulfilled1", "namely1", "flag1", "allegedly1", "wittingly1",
                                  "unwittingly1", "sanchez1", "allegation1", "volume1", "emphasis1", "hysterical1",
                                  "occasion1", "observe1", "jacob1", "queue1", "mitchell1", "logic1", "falsehood1",
                                  "correctly1", "description1", "factual1", "factually1", "incorrect1", "theory1",
                                  "aftermath1", "myth1", "debunk1", "rand1", "documentation1", "counter1", "shortly1",
                                  "defending1", "whirl1", "evacuate1", "distressing1", "congress1", "indicate1",
                                  "directed1", "deck1", "nowhere1", "cairo1", "victoria1", "spokesperson1", "spray1",
                                  "demonstration1", "eyewitness1", "rice1", "consequence1", "spontaneous1", "false1",
                                  "reservation1", "narrative1", "roof1", "grave1", "egyptian1", "inflammatory1",
                                  "attacker1", "justification1", "tunis1", "thankfully1", "appreciate1", "pressed1",
                                  "approximately1", "fluid1", "affiliated1", "conflicting1", "slide1", "rhodes1",
                                  "assume1", "alive1", "square1", "chapter1", "disservice1", "glad1", "confusing1",
                                  "motivated1", "khartoum1", "burning1", "tunisian1", "desperate1", "insinuation1",
                                  "plain1", "vastly1", "monitor1", "sake1", "scheme1", "entitled1", "premium1",
                                  "documentary1", "buck1", "exhibit1", "ceiling1", "recitation1", "sheer1", "metric1",
                                  "regularity1", "amazing1", "centric1", "aired1", "cancel1", "conclusion1",
                                  "investigator1", "ignore1", "damn1", "singly1", "briefly1", "merit1", "confused1",
                                  "accusation1", "sleep1", "wrack1", "brain1", "predecessor1", "nairobi1", "alike1",
                                  "east1", "optimist1", "prosecution1", "stack1", "sad1", "regardless1", "undisputed1",
                                  "bubble1", "solicit1", "redact1", "identifier1", "insight1", "keeper1", "tomorrow1",
                                  "intervene1", "electricity1", "definition1", "evaluate1", "print1", "determination1",
                                  "somewhere1", "thread1", "prolific1", "explanation1", "tyler1", "rhetoric1",
                                  "derision1", "seizing1", "jaw1", "unenthusiastic1", "paralysis1", "lukewarm1",
                                  "operational1", "pseudo1", "venture1", "madam1", "drivel1", "eminently1",
                                  "relevance1", "classification1", "subpoena1", "deposition1", "compelling1",
                                  "unilaterally1", "marshal1", "sir1", "morse1", "carrier1", "pigeon1", "smoke1",
                                  "irrelevant1", "telephone1", "bothered1", "motion1", "parliamentarian1", "mills1",
                                  "fairness1", "discovery1", "odd1", "outstanding1", "admission1", "selectively1",
                                  "indulgence1", "signify1", "clerk1", "adequately1", "soft1", "mailing1", "react1",
                                  "favorable1", "avenue1", "best1", "newly1", "lever1", "register1", "follower1",
                                  "incoming1", "adamantly1", "convey1", "engage1", "messaging1", "folder1", "medicine1",
                                  "gasoline1", "diesel1", "milk1", "visa1", "basically1", "affirmatively1",
                                  "positively1", "fuel1", "insult1", "servant1", "similarly1", "elements1",
                                  "unofficial1", "originally1", "buttonhole1", "reception1", "helpful1", "carefully1",
                                  "incredulity1", "testify1", "aisle1", "overwhelm1", "somehow1", "posed1",
                                  "historical1", "influential1", "deference1", "consent1", "reopen1", "full-time1",
                                  "sentiment1", "inhabit1", "permanent1", "promote1", "vibrant1", "bilateral1",
                                  "outweigh1", "elimination1", "profoundly1", "lacking1", "minimal1", "nonexistent1",
                                  "configuration1", "geography1", "mid-july1", "formally1", "hiding1", "headquarters1",
                                  "accurate1", "duration1", "dispute1", "predominantly1", "located1", "informative1",
                                  "update1", "gradual1", "deem1", "consul1", "protocol1", "accurately1", "disclosed1",
                                  "assignment1", "militia1", "disarm1", "miss1", "dated1", "universe1", "curious1",
                                  "humor1", "typing1", "picking1", "spirit1", "entrepreneurial1", "barrier1", "length1",
                                  "describe1", "patrick1", "high-level1", "evaluation1", "advised1", "abandon1",
                                  "contrary1", "unaware1", "prioritize1", "farthest1", "unquestionably1", "viewer1",
                                  "embarrassment1", "reference1", "prosecute1", "propel1", "passionate1", "rip1",
                                  "remove1", "juxtapose1", "difficulty1", "drag1", "atmosphere1", "depth1", "specify1",
                                  "undergo1", "circle1", "aunt1", "remainder1", "haven1", "inflict1", "inhalation1",
                                  "enable1", "endeavor1", "interior1", "proceeding1", "instinct1", "desperately1",
                                  "saving1", "succumb1", "horror1", "resuscitate1", "labored1", "fog1", "heroism1",
                                  "professionalism1", "refuge1", "fortified1", "whisper1", "fighter1", "god1",
                                  "one-tenth1", "row1", "forbid1", "soliloquy1", "elegant1", "finger1", "tape1",
                                  "rewind1", "regret1", "violation1", "cell1", "24/71", "accordance1", "respects1",
                                  "breakfast1", "trick1", "follow-up1", "inadequate1", "grossly1", "magnitude1",
                                  "preventable1", "premises1", "vulnerable1", "lease1", "consensus1", "west1",
                                  "professionalize1", "height1", "province1", "requirement1", "dip1", "dialogue1",
                                  "boring1", "pedigree1", "honorable1", "malign1", "sic1", "retreated1", "accountable1",
                                  "oversight1", "mismanagement1", "price1", "inflexible1", "limb1", "armor1", "static1",
                                  "kinetic1", "flexibility1", "vet1", "reliable1", "invite1", "friendly1", "accompany1",
                                  "landed1", "firearm1", "vienna1", "trigger1", "funded1", "responsiveness1", "echo1",
                                  "expression1", "havoc1", "causing1", "therefore1", "explicit1", "berlin1", "rome1",
                                  "istanbul1", "appoint1", "laying1", "foundation1", "timeline1", "bother1", "key1",
                                  "explicitly1", "comprehensive1", "excerpt1", "cabinet1", "instruct1", "recognition1",
                                  "vacation1", "handed1", "guarantee1", "driveway1", "timing1", "suggestion1",
                                  "rescue1", "import1", "insofar1", "doctrine1", "publicize1", "coalition1",
                                  "sanction1", "imposed1", "disposal1", "extent1", "lap1", "precise1", "recess1",
                                  "mess1", "bolster1", "approve1", "approving1", "premise1", "distinction1",
                                  "listening1", "chaos1", "spike1", "cross4", "forthcoming1", "monitoring1",
                                  "consideration1", "departure1", "ultimate1", "oversee1", "properly1", "fault1",
                                  "mislead1", "pleased1", "beef1", "maximum1", "informing1", "tourist1", "criterion1",
                                  "extreme1", "conditioning1", "orientation1", "imperative1", "gavel1", "quicker1",
                                  "attached1", "run-up1", "append1", "hampshire1", "asset1", "spinning1", "suspicion1",
                                  "leon1", "undeniable1", "intentionally1", "pale1", "surveillance1", "drone1",
                                  "ordering1", "rota1", "spain1", "croatia1", "mobilize1", "redirect1", "bound1",
                                  "dispatch1", "satisfied1", "dissipated1", "actively1", "assertion1", "scrambled1",
                                  "logistics1", "upheaval1", "volatility1", "amateur1", "mock1", "afghan1", "tire1",
                                  "indonesian1", "jakarta1", "volatile1", "prayer1", "battering1", "army1", "fancy1"};

double EPSILON = 0.000001;




/**
 * Helper for to generate test data of size.
 * @param n number of elements in the test.
 * @return a json struct in type of a graph code.
 */
json generateTestData(int n) {
    nlohmann::json gcq;

    DICT.insert(DICT.end(), DICT2.begin(), DICT2.end());

    if (n > DICT.size()) {
        exit(71);
    }

    std::vector<std::string> newVec(DICT.begin(), DICT.begin() + n);
    //extractElements(DICT, subA, n);

    std::vector<std::vector<int>> data;
    for (int i = 0; i < n; i++) {
        std::vector<int> x;
        data.push_back(x);
        for (int j = 0; j < n; j++) {

            if (i == j) {
                //data[i][j] = 1;
                data[i].push_back(1);
            } else {
                data[i].push_back(i + j % 2);
            }
        }
    }

    gcq["dictionary"] = newVec;
    gcq["matrix"] = data;
    return gcq;
}

/**
 * Test helper to compare double values.
 * @param a comperand
 * @param b comparand
 * @return true if equal unti limit of EPSILON.
 */
bool AreSame(double a, double b) {
    return fabs(a - b) < EPSILON;
}

/**
 * Generated a Graph Code for testing.
 *
 * @param dimension the dimension of the word list and matrix
 * @return the generated Graph Code
 */
GraphCode generateTestDataGc(int dimension) {

    if (dimension > DICT.size()) {

        int count = 0;
        for(int i = 0; count < dimension; i++) {
            for (std::string str: DICT2) {
                DICT.push_back(str + std::to_string(i));
                count++;
            }

        }
    }

    std::vector<std::string> *newVec = new std::vector<std::string>(DICT.begin(), DICT.begin() + dimension);
    unsigned short *data = (unsigned short *) malloc(sizeof(int) * dimension * dimension);

    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {

            if (i == j) {
                data[i * dimension + j] = 1;
            } else {
                data[i * dimension + j] = i + j % 2;
            }
        }
    }

    GraphCode gc;
    gc.dict = newVec;
    gc.matrix = data;

    return gc;
}


