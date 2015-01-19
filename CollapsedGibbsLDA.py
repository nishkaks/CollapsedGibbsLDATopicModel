#-------------------------------------------------------------------------------
# Name:        CollapsedGibbsLDA
# Purpose:
#
# Author:      Nishka K.S
#
# Created:     26/11/2014
# Copyright:   (c) Nishka 2014
#-------------------------------------------------------------------------------

#import scipy;
import numpy;
import nltk;
import re;
from gensim import corpora;
import optparse;

# Function to load vocabulary from a subset of documents in the brown corpus.
def load_vocabulary():

    from nltk.corpus import brown;

    # select a subset of documents from brown corpus
##    fileIDS = ['cd01', 'cd02', 'cd03','cd04','cd05','cd06','cd07','cd08',   # religion
##               'ca01', 'ca02', 'ca03','ca04', 'ca05', 'ca06','ca07', 'ca08', # political new reporting
##               'cj01', 'cj02','cj03','cj04','cj05','cj06','cj07','cj18','cj19']; # science
##               #'cl04', 'cl11','cl17','cl19','cl20','cl22','cl23','cl24']; # murder mystery

    fileIDS = ['cd01', 'cd02', 'cd03','cd09','cd11','cd14','cd16','cd10','cd04','cd05',   # religion
               'ca01', 'ca02', 'ca03','ca04', 'ca05', 'ca06','ca07', 'ca08','ca09','ca10', # political news reporting
               'cj01', 'cj02','cj03','cj04','cj05','cj06','cj07','cj18','cj19','cj20']; # science


    #stopword_list = nltk.corpus.stopwords.words('english');
    stopword_list = "a,s,give,made,af,make,able,about,above,according,accordingly,across,actually,after,afterwards,again,against,ain,t,all,allow,allows,almost,alone,along,already,also,although,always,am,among,amongst,an,and,another,any,anybody,anyhow,anyone,anything,anyway,anyways,anywhere,apart,appear,appreciate,appropriate,are,aren,t,around,as,aside,ask,asking,associated,at,available,away,awfully,be,became,because,become,becomes,becoming,been,before,beforehand,behind,being,believe,below,beside,besides,best,better,between,beyond,both,brief,but,by,c,mon,c,s,came,can,can,t,cannot,cant,cause,causes,certain,certainly,changes,clearly,co,com,come,comes,concerning,consequently,consider,considering,contain,containing,contains,corresponding,could,couldn,t,course,currently,definitely,described,despite,did,didn,t,different,do,does,doesn,t,doing,don,t,done,down,downwards,during,each,edu,eg,eight,either,else,elsewhere,enough,entirely,especially,et,etc,even,ever,every,everybody,everyone,everything,everywhere,ex,exactly,example,except,far,few,fifth,first,five,followed,following,follows,for,former,formerly,forth,four,from,further,furthermore,get,gets,getting,given,gives,go,goes,going,gone,got,gotten,greetings,had,hadn,t,happens,hardly,has,hasn,t,have,haven,t,having,he,he,s,hello,help,hence,her,here,here,s,hereafter,hereby,herein,hereupon,hers,herself,hi,him,himself,his,hither,hopefully,how,howbeit,however,i,d,i,ll,i,m,i,ve,ie,if,ignored,immediate,in,inasmuch,inc,indeed,indicate,indicated,indicates,inner,insofar,instead,into,inward,is,isn,t,it,it,d,it,ll,it,s,its,itself,just,keep,keeps,kept,know,knows,known,last,lately,later,latter,latterly,least,less,lest,let,let,s,like,liked,likely,little,look,looking,looks,ltd,mainly,many,may,maybe,me,mean,meanwhile,merely,might,more,moreover,most,mostly,much,must,my,myself,name,namely,nd,near,nearly,necessary,need,needs,neither,never,nevertheless,new,next,nine,no,nobody,non,none,noone,nor,normally,not,nothing,novel,now,nowhere,obviously,of,off,often,oh,ok,okay,old,on,once,one,ones,only,onto,or,other,others,otherwise,ought,our,ours,ourselves,out,outside,over,overall,own,particular,particularly,per,perhaps,placed,please,plus,possible,presumably,probably,provides,que,quite,qv,rather,rd,re,really,reasonably,regarding,regardless,regards,relatively,respectively,right,said,same,saw,say,saying,says,second,secondly,see,seeing,seem,seemed,seeming,seems,seen,self,selves,sensible,sent,serious,seriously,seven,several,shall,she,should,shouldn,t,since,six,so,some,somebody,somehow,someone,something,sometime,sometimes,somewhat,somewhere,soon,sorry,specified,specify,specifying,still,sub,such,sup,sure,t,s,take,taken,tell,tends,th,than,thank,thanks,thanx,that,that,s,thats,the,their,theirs,them,themselves,then,thence,there,there,s,thereafter,thereby,therefore,therein,theres,thereupon,these,they,they,d,they,ll,they,re,they,ve,think,third,this,thorough,thoroughly,those,though,three,through,throughout,thru,thus,to,together,too,took,toward,towards,tried,tries,truly,try,trying,twice,two,un,under,unfortunately,unless,unlikely,until,unto,up,upon,us,use,used,useful,uses,using,usually,value,various,very,via,viz,vs,want,wants,was,wasn,t,way,we,we,d,we,ll,we,re,we,ve,welcome,well,went,were,weren,t,what,what,s,whatever,when,whence,whenever,where,where,s,whereafter,whereas,whereby,wherein,whereupon,wherever,whether,which,while,whither,who,who,s,whoever,whole,whom,whose,why,will,willing,wish,with,within,without,won,t,wonder,would,would,wouldn,t,yes,yet,you,you,d,you,ll,you,re,you,ve,your,yours,yourself,yourselves,zero".split(',')
    #print stopword_list
    lemma = nltk.WordNetLemmatizer();

    # Read corpus, lemmatize, remove stop words and punctuation
    texts = [[lemma.lemmatize(word.lower()) for word in brown.words(fileID) if (word.lower() not in stopword_list) and re.match(r'[a-zA-Z]+$',word)] for fileID in fileIDS]
    #texts = [[word.lower() for word in brown.words(fileID) if (word.lower() not in stopword_list) and re.match(r'[a-zA-Z]+$',word)] for fileID in fileIDS]

    # remove words that appear only once
    all_tokens = sum(texts, []);
    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1);
    texts = [[word for word in text if word not in tokens_once] for text in texts];

    return texts;

def main():
    parser = optparse.OptionParser();
    parser.add_option("--alpha", dest="alpha", type = "float", help="hyper parameter alpha - document-topic");
    parser.add_option("--beta", dest="beta", type = "float", help="hyper parameter beta - topic-word");
    parser.add_option("-T", dest="T", type = "int", help="number of topics");
    parser.add_option("-i", dest="iterations", type = "int", help="number of iterations");
    (options,args) = parser.parse_args();
    if not (options.alpha or options.beta or options.T):
        parser.error("Please specify all the parameters alpha, beta and T");

    # Load vocabulary from a subset of documents in the brown corpus.
    Vocab = load_vocabulary();
    #print(len(Vocab)); #number of documents

    # use gensim to create word IDs, dictionary and counts
    dictionary = corpora.Dictionary(Vocab);
    dictionary.save('E:/Study/Fall2014/CS760/brown.dict') # store the dictionary, for future reference
    wordID = dictionary.token2id;
    wordtoken = dict((id, token) for token, id in wordID.iteritems())
    wordDocFreq = [dictionary.doc2bow(text) for text in Vocab];
    corpora.MmCorpus.serialize('E:/Study/Fall2014/CS760/brown.mm', wordDocFreq) # store to disk, for later use
    #print(wordDocFreq);
    #print(wordID);
    #print(len(wordID)); # number of words in vocab

    # initialize lists
    iterations = options.iterations;
    W = len(wordID); # numbers of words in vocab
    D = len(Vocab); # number of documents
    beta = options.beta;
    T = options.T;
    alpha = options.alpha;
    C_WT = numpy.zeros((W,T)) + beta; # number of time word w is assigned to topic j, not including current instances of i
    C_T = numpy.zeros(T) + W * beta; # number of words in each topic
    C_TD = numpy.zeros((T,D)) + alpha; # number of times topic j is assigned to some word token in document d, not including current instances of i
    C_D = numpy.zeros(D) + T * alpha;

    Z_D_WT = [];

    # randomly assign topics to words and update counts
    for i, document in enumerate(wordDocFreq):
        Z_WT = []
        for t,word in enumerate(document):
            #print(i,word[0],word[1]);
            j = numpy.random.randint(0,T);
            C_TD[j,i] +=1;
            C_WT[word[0],j] +=1;
            C_T[j] +=1;
            C_D[i] +=1;
            Z_WT.append(j);
        Z_D_WT.append(numpy.array(Z_WT));

    # Collapsed gibbs sampling
    for i in range(iterations):
        for di, document in enumerate(wordDocFreq):
            for t,word in enumerate(document):
                # Decrement values
                z = Z_D_WT[di][t];
                C_TD[z,di] -=1;
                C_WT[word[0],z] -=1;
                C_T[z] -=1;
                C_D[di] -=1;

                # estimate P(Z | ....)
                p_z = C_WT[word[0],:] * C_TD[:,di]/((C_T) * C_D[di]);
                p_z /= numpy.sum(p_z);
                #print p_z

                # sample new topic z from multinomial
                z = numpy.random.multinomial(1,p_z).argmax();

                # Update the values
                Z_D_WT[di][t] = z;
                C_TD[z,di] +=1;
                C_WT[word[0],z] +=1;
                C_T[z] +=1;
                C_D[di] +=1;

    # Compute theta
    theta = C_TD + alpha;
    theta /=numpy.sum(theta,axis=0)
    # Compute phi
    phi = C_WT + beta;
    phi /= numpy.sum(phi,axis=0)
    phisorted = numpy.argsort(phi,axis=0);
    #print wordtoken


    phisorted = phisorted[numpy.shape(phisorted)[0]-20:numpy.shape(phisorted)[0],:]
    for i in xrange(T):
        print i
        for j in xrange(20):
            print wordtoken.get(phisorted[19-j,i]);





##    print numpy.sum(phi,axis=0)
##    print numpy.sum(theta,axis=0)
##    print numpy.shape(theta),numpy.shape(phi),W
##    print numpy.shape(C_TD),numpy.shape(C_WT)

    # triangle

    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x = np.linspace(0, 1, 100)
    y = 1-x
    ax.plot(x, y, zs=0, zdir='z',c='b')

    y = np.linspace(0, 1, 100)
    z = 1-y
    x=np.zeros(len(y))
    ax.plot(x, y, z,c='b')

    z = np.linspace(0,1,100)
    x=1-z
    y =np.zeros(len(z))
    ax.plot(x,y,z,c='b')
    x= theta[0]
    y= theta[1]
    z=theta[2]

    ax.plot(x[0:9], y[0:9], z[0:9],'s', c='m', label='religion')
    ax.plot(x[10:19], y[10:19], z[10:19],'o', c='g', label='politics')
    ax.plot(x[20:29], y[20:29], z[20:29],'*', c='c', label='science')

    ax.legend()
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)

    plt.show()


if __name__ == '__main__':
    main()
