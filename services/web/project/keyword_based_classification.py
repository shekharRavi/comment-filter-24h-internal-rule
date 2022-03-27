import pandas as pd
import re
from lingua import Language, LanguageDetectorBuilder

from langdetect import detect
from tqdm import tqdm

lang_detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
valid_langs= ['BOSNIAN', 'SERBIAN','CROATIAN','MONTENEGRIN','ENGLISH', 'SLOVENE','SLOVAK', 'hr','sr','bs','cnr','en','sl','slv']

dir_name = '/import/cogsci/ravi/datasets/24sata/'
out_dir = dir_name +'word_counts/'

r2_file = '2_possible_words.csv'
r3_file = '3_possible_words.csv'
r4_file = '4_possible_words.csv'
r5_file = '5_possible_words.csv'
r6_file = '6_possible_words.csv'

R2=pd.read_csv(out_dir+r2_file).rule_word.tolist()
R3=pd.read_csv(out_dir+r3_file).rule_word.tolist()
R4=pd.read_csv(out_dir+r4_file).rule_word.tolist()
R5=pd.read_csv(out_dir+r5_file).rule_word.tolist()
R6=pd.read_csv(out_dir+r6_file).rule_word.tolist()

not_allowed_words = ['24sata', 'admin', 'Adolf', 'Hitler', 'jebem', 'jebemti', 'pička', 'lezba', 'majmun',
                     'mater', 'majku', 'kurac', 'seri', 'seres', 'nabijem', 'glup', 'idijot', 'idiot', 'budala',
                     'kreten', 'pizda', 'klad', 'kladi', 'ZDS', 'klaunski', 'mater', 'smece', 'tenkre', 'bilde',
                     'debil', 'jebi', 'cigan', 'govn', 'seljac', 'drolj', 'kozojeb', 'musliman', 'klaunski',
                     'derpe', 'maloumni', 'hebe', 'racku', 'spodob', 'kita', 'stoka', 'crkn', 'debel', 'krmaca',
                     'dubre', 'djubre', 'retard', 'barbika', 'miss piggy', 'srb', 'kme', 'novina', 'pick',
                     'nepism', 'sipt', 'ptar', 'za dom', 'srps', 'tuka', 'jeb, peni', 'udba, čas', 'šiptar',
                     'šupak', 'ustaša', 'smrad', 'budal', 'kopile', 'imbecil', 'guba', 'stoko', 'icka', 'ubre',
                     'gnjida', 'ljubičice', 'štraca', 'šljam', 'pupavac', 'jado', 'bilde', 'straca', 'sereš',
                     'đubre', 'čedo', 'pička', 'jadnič', 'četn', 'kurc', 'krmača', 'lomač', 'metak', 'čelo',
                     'jebo', 'ubit', 'asenovac', 'cetni', 'gamad', 'kurva', 'peder', 'kurvetina', 'kurv',
                     'kobilo', 'degenerik', 'panglu', 'tenkre', 'smeće', 'sme', 'smece', 'sponzoruša', 'sponzorusa',
                     'sponz', 'konju', 'krivousti', 'krivou', 'hanzek', 'hanžek', 'lešinari', 'lesinari',
                     'ološ', 'olos', 'papcino', 'papak', 'papčino', 'bosanko', 'bosanđeros', 'bosanceros',
                     'bosancina', 'hercegovance', 'šupak', 'šupci', 'supak', 'supci', 'bosančeros', 'kuja',
                     'kujica', 'dementra', 'dementna', 'nakaza', 'katolibani', 'talibani', 'papčina', 'kuraba',
                     'ganci', 'ljadro', 'retard', 'paksu', 'droca', 'express', 'srba', 'srbi', 'expres' ,
                     'šuft', 'suft', 'ćifut', 'katoliban', 'kolje', 'klati', 'kolj', 'jambrusic', 'jambrušić',
                     'tolusic', 'tolušić', 'siptar', 'balija', 'droca', 'acab', 'a.c.a.b.', 'radman',
                     'selekcija', 'sjajna zvijezdo', 'sjajna zvjezdo', 'celofanka', 'kravo', 'kobila',
                     'samoprozvani','doktor za ljubav', 'drkolinda', 'poturica', 'poturico', 'isprdak']

# word_to_rule=[]

def check_blocked_words(text,not_allowed_words=[]):
    # Simply based on the texonomy
    rule_flag = False
    for word in not_allowed_words:
        match = re.search(word, text)
        if match:
            rule_flag = True
            # found_word = word  # TODO: This misses if multiple words are present
            break
    return rule_flag

def check_rule_seven(text):
    # To check based on the language and upper case
    rule_flag = False
    if text.isupper():
        rule_flag = True
        print('All upper')
    else:
        text = " ".join(re.findall("[a-zA-Z]+", text))
        words = text.split(' ')
        c_w = len(words)
        if c_w > 3:
            rule_flag = True
            # Check in two instances
            val1 = " ".join(words[:int(c_w/2)])
            val2 = " ".join(words[int(c_w/2):])
            for val in [val1, val2,text]:
                try:
                    lang2 = detect(val) #Easy to run
                except:
                    lang2 = 'hr'
                # print(lang2)
                if lang2 not in valid_langs:
                    try:
                        lang1 = str(lang_detector.detect_language_of(val))[9:]
                    except:
                        lang1 = 'CROATIAN'

                    if lang1 in valid_langs: #Found valid language, so not breaking Rule 7
                        rule_flag = False  # Found valid language, so not breaking Rule 7
                        break
                else:
                    rule_flag = False #Found valid language, so not breaking Rule 7
                    break

    return rule_flag


def keyword_to_rule(text,rule_words,threshold=2):
    words = text.split(' ')
    c = 0
    rule_flag = False
    for word in words:
        if word in rule_words:
            c +=1
            if c > threshold:
                rule_flag = True #TODO: use overall ratio to determine instead of count only
                break
    return rule_flag

def keyword_based_classification(text):
    #TODO: we coud include keyword based for the Rule 1 and 8 also.
    rule = -1
    conf = -1
    #Check blacklisted words
    rule_flag = check_blocked_words(text, not_allowed_words=not_allowed_words)
    if rule_flag:
        rule = 3
        conf = 0.7
    else:
        #Check rule 7 based on language or all Caps
        rule_flag = check_rule_seven(text)

        if rule_flag:
            rule = 7
            conf = 0.9
        else:
            #Check based on keywords for Rule 2,3,4,5,6
            rule_words = [R2, R3,R4,R5,R6]
            thresholds = [2,2,2,1,1] #For Major rule higher threshold
            rules = [2,3,4,5,6]
            for rule_word, threshold, rule in zip(rule_words,thresholds, rules):
                rule_flag = keyword_to_rule(text,rule_words,threshold=threshold)
                if rule_flag:
                    rule = rule
                    conf = 0.7
                    break
    return rule, conf

def proces_file():

    file_name = 'csv_embeddia_export.csv'

    df = pd.read_csv(dir_name+ file_name)

    df['key_rule']=0

    for i in tqdm(range(len(df))):
        content = df.loc[i,'content']

        rule, conf = keyword_based_classification(content)

        if rule != -1:
            df.loc[i, 'key_rule'] = rule

        if i > 100:
            break

    save_file = out_dir+'keywords_rule_'+file_name
    df.to_csv(save_file, index=False)
    print('Saved to', save_file)

proces_file()