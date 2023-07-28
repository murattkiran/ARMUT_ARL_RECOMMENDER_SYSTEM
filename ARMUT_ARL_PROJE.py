
#########################
# İş Problemi
#########################

# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning (Birliktelik Kuralı Öğrenimi) ile ürün tavsiye sistemi oluşturulmak istenmektedir.


#########################
# Veri Seti
#########################
#Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih




#########################
# GÖREV 1: Veriyi Hazırlama
#########################

import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

"""Apriori algoritması adım adım çalışmanın başında belirlenecek bir Support eşik değerine göre olası ürün 
çiftlerini hesaplar ve her iterasyonda belirlenen Support değerine göre elemeler yaparak nihai final 
tablosunu oluşturur.
"""


# Adım 1: armut_data.csv dosyasınız okutunuz.

df_ = pd.read_csv("datasets/armut_data.csv")
df = df_.copy()

df.head()
#    UserId  ServiceId  CategoryId           CreateDate
# 0   25446          4           5  2017-08-06 16:11:00
# 1   22948         48           5  2017-08-06 16:12:00
# 2   10618          0           8  2017-08-06 16:13:00
# 3    7256          9           4  2017-08-06 16:14:00
# 4   25446         48           5  2017-08-06 16:16:00

# Çıktı aşağıya doğru kalabalık bir şekilde gelmedi çünkü pd.set_option('display.width', 500) kullandık.
# Çıktı tek bir satırda geldi çünkü pd.set_option('display.expand_frame_repr', False) kullandık.


df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 162523 entries, 0 to 162522
# Data columns (total 4 columns):
#  #   Column      Non-Null Count   Dtype
# ---  ------      --------------   -----
#  0   UserId      162523 non-null  int64
#  1   ServiceId   162523 non-null  int64
#  2   CategoryId  162523 non-null  int64
#  3   CreateDate  162523 non-null  object
# dtypes: int64(3), object(1)


df.describe().T
#                count          mean          std  min     25%      50%      75%      max
# UserId      162523.0  13089.803862  7325.816060  0.0  6953.0  13139.0  19396.0  25744.0
# ServiceId   162523.0     21.641140    13.774405  0.0    13.0     18.0     32.0     49.0
# CategoryId  162523.0      4.325917     3.129292  0.0     1.0      4.0      6.0     11.0

# Bu dataframe'i betimlediğimizde çıkan sonuçlar mantıklı gözükmektedir.


df.isnull().sum()
# UserId        0
# ServiceId     0
# CategoryId    0
# CreateDate    0
# dtype: int64

# Veri setinde eksik değer olup olmadığını kontrolünü yaptık ve olmadığını gözlemledik.


df.shape
# (162523, 4)

# df.shape diyerek kaç gözlem olduğuna baktık. 162523 gözlem var.

df["UserId"].nunique()


# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID'yi "_" ile birleştirerek hizmetleri temsil edecek yeni bir değişken oluşturunuz.

df["Hizmet"] = df["ServiceId"].astype(str) + '_' + df["CategoryId"].astype(str)

df.head()
#    UserId  ServiceId  CategoryId           CreateDate Hizmet
# 0   25446          4           5  2017-08-06 16:11:00    4_5
# 1   22948         48           5  2017-08-06 16:12:00   48_5
# 2   10618          0           8  2017-08-06 16:13:00    0_8
# 3    7256          9           4  2017-08-06 16:14:00    9_4
# 4   25446         48           5  2017-08-06 16:16:00   48_5




# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning (Birliktelik Kuralı Öğrenimi) uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir. Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti;
# 2017’in 10.ayında aldığı  9_4, 38_4  hizmetleri başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_"
# ile birleştirirek ID adında yeni bir değişkene atayınız.

# İlk olarak CreateDate sütununu datetime formatına dönüştürelim:

df["CreateDate"] = pd.to_datetime(df["CreateDate"])

# Kontrol edelim:
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 162523 entries, 0 to 162522
# Data columns (total 5 columns):
#  #   Column      Non-Null Count   Dtype
# ---  ------      --------------   -----
# 0   UserId      162523 non-null  int64
# 1   ServiceId   162523 non-null  int64
# 2   CategoryId  162523 non-null  int64
# 3   CreateDate  162523 non-null  datetime64[ns]
# 4   Hizmet      162523 non-null  object

# New_Date isminde sadece yıl ve ay içeren yeni bir değişken oluşturalım:

df["New_Date"] = df["CreateDate"].dt.to_period("M")

# dt.to_period metodu, pandas Serilerindeki veya DataFrame sütunlarındaki
# tarih veya zaman bilgisini dönemlere dönüştürmek için kullanılır.
# Y" yıllık dönemler, "M" aylık dönemler, "D" günlük dönemler gibi sıklıklar kullanılabilir.
# Dönüştürmek istediğiniz zaman aralığına bağlı olarak farklı sıklıklar kullanabilirsiniz.


df.head()
#    UserId  ServiceId  CategoryId          CreateDate Hizmet New_Date
# 0   25446          4           5 2017-08-06 16:11:00    4_5  2017-08
# 1   22948         48           5 2017-08-06 16:12:00   48_5  2017-08
# 2   10618          0           8 2017-08-06 16:13:00    0_8  2017-08
# 3    7256          9           4 2017-08-06 16:14:00    9_4  2017-08
# 4   25446         48           5 2017-08-06 16:16:00   48_5  2017-08


# Şimdi UserID ve New_Date değişkenlerini "_" ile birleştirerek "SepetID" adında yeni bir değişkene atayalım:

df["SepetID"] = df["UserId"].astype(str) + '_' + df["New_Date"].astype(str)

df.head()
#    UserId  ServiceId  CategoryId          CreateDate Hizmet New_Date        SepetID
# 0   25446          4           5 2017-08-06 16:11:00    4_5  2017-08  25446_2017-08
# 1   22948         48           5 2017-08-06 16:12:00   48_5  2017-08  22948_2017-08
# 2   10618          0           8 2017-08-06 16:13:00    0_8  2017-08  10618_2017-08
# 3    7256          9           4 2017-08-06 16:14:00    9_4  2017-08   7256_2017-08
# 4   25446         48           5 2017-08-06 16:16:00   48_5  2017-08  25446_2017-08

df["SepetID"].nunique()


#########################
# GÖREV 2: Birliktelik Kuralları Üretiniz ve Öneride bulununuz
#########################

# Adım 1: Aşağıdaki gibi sepet hizmet pivot table’i oluşturunuz.

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..


# Adım adım gidelim:
# Önce SepetID'ye göre kır, sonra Hizmet'lere göre kır, bu sepette bak bakalım satın alınmış Hizmet'ten kaç tane var demek için:
# CategoryId zaten Hizmet'in içinde dolayısıyla Hizmet'in count unu da alabiliriz, aynı şey.

df.groupby(["SepetID", "Hizmet"]).agg({"CategoryId": "count"}).unstack().iloc[0:5, 0:6]
#               CategoryId
# Hizmet               0_8 10_9 11_11 12_7 13_11 14_7
# SepetID
# 0_2017-08            NaN  NaN   NaN  NaN   NaN  NaN
# 0_2017-09            NaN  NaN   NaN  NaN   NaN  NaN
# 0_2018-01            NaN  NaN   NaN  NaN   NaN  NaN
# 0_2018-04            NaN  NaN   NaN  NaN   NaN  1.0
# 10000_2017-08        NaN  NaN   NaN  NaN   NaN  NaN

# unstack: groupby işleminden sonra pivot yapmak için yani "Hizmet" değişkenini sütunlara geçirmek için unstack() fonsiyonunu kullanıyoruz.
# Yukarıda görüleceği üzere boş olan yerlere NA, eğer bir Hizmet satın alma durumu varsa bunların sayıları geldi.
# Bir sorunumuz daha var, biz buralarda eksik değerlerin yerinde 0, dolularda 1 yazsın istiyoruz.

# unstack() işleminden sonra boş olan yerleri 0 ile doldurmak için fillna(0) dediğimizde boşluklar 0 ile dolmuş olacak.

df.groupby(["SepetID", "Hizmet"]).agg({"CategoryId": "count"}).unstack().fillna(0).iloc[0:5, 0:6]
#               CategoryId
# Hizmet               0_8 10_9 11_11 12_7 13_11 14_7
# SepetID
# 0_2017-08            0.0  0.0   0.0  0.0   0.0  0.0
# 0_2017-09            0.0  0.0   0.0  0.0   0.0  0.0
# 0_2018-01            0.0  0.0   0.0  0.0   0.0  0.0
# 0_2018-04            0.0  0.0   0.0  0.0   0.0  1.0
# 10000_2017-08        0.0  0.0   0.0  0.0   0.0  0.0

# Burada 1 yazan yerde kimi durumda daha büyük sayılar da yazabilir
# 0'dan büyük herhangi bir sayıya 1 yazılması lazım diğerlerinde 0 yazması lazım.(binary encode)
# Çünkü daha ölçülebilir üzerinde analitik işlemler yapabileceğimiz özel bir matris yapısı bekliyoruz.
# Bunun için bütün gözlemleri gezen (satır ve sütunların hepsinde) applymap() fonksiyonunu kullanacağız:

df.groupby(["SepetID", "Hizmet"]). \
    agg({"CategoryId": "count"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:6]
#               CategoryId
# Hizmet               0_8 10_9 11_11 12_7 13_11 14_7
# SepetID
# 0_2017-08              0    0     0    0     0    0
# 0_2017-09              0    0     0    0     0    0
# 0_2018-01              0    0     0    0     0    0
# 0_2018-04              0    0     0    0     0    1
# 10000_2017-08          0    0     0    0     0    0


# Sonuç olarak:
invoice_product_df = df.groupby(["SepetID", "Hizmet"])["CategoryId"].count().unstack().fillna(0).\
    applymap(lambda x: 1 if x > 0 else 0)




# Adım 2: Birliktelik kurallarını oluşturunuz.

# İlk olarak apriori() fonksiyonu ile olası tüm ürün birlikteliklerinin Support değerlerini yani olasılıklarını bulacağız.
# Burada min_support, belirlemek istediğimiz minimum Support değeri, eşik değeri
# Kullanmak istediğimiz veri setindeki değişkenlerin isimlerini kullanmak istiyorsak use_colnames=True yaparız.

frequent_itemsets = apriori(invoice_product_df.astype("bool"),
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)
#      support      itemsets
# 8   0.238121        (18_4)
# 19  0.130286         (2_0)
# 5   0.120963        (15_1)
# 39  0.067762        (49_1)
# 28  0.066568        (38_4)
# 3   0.056627       (13_11)
# 12  0.047515        (22_0)
# 9   0.045563        (19_6)
# 15  0.042895        (25_0)
# 7   0.041533        (17_5)
# 45  0.041393         (9_4)
# 42  0.040284         (6_7)
# 38  0.037518        (48_5)
# 41  0.036408        (5_11)
# 40  0.035875         (4_5)
# 47  0.033951   (15_1, 2_0)
# 2   0.029374        (12_7)
# 37  0.028756        (47_7)
# 13  0.027577       (23_10)
# 16  0.027492        (26_7)
# 23  0.027310        (33_4)
# 18  0.026580        (29_0)
# 1   0.026523       (11_11)
# 22  0.026032        (32_4)
# 14  0.024038       (24_10)
# 4   0.023406        (14_7)
# 30  0.022072         (3_5)
# 11  0.020963        (20_5)
# 21  0.020921        (31_6)
# 36  0.020051        (46_4)
# 25  0.020008       (35_11)
# 0   0.019728         (0_8)
# 10  0.019292         (1_4)
# 24  0.019194        (34_6)
# 27  0.018394        (37_0)
# 17  0.017397        (27_7)
# 29  0.016933       (39_10)
# 52  0.016568   (22_0, 2_0)
# 31  0.015922        (40_8)
# 33  0.015782        (43_2)
# 32  0.015206        (41_3)
# 43  0.015024         (7_3)
# 6   0.014659        (16_8)
# 34  0.014153        (44_0)
# 53  0.013437   (2_0, 25_0)
# 46  0.012819  (2_0, 13_11)
# 35  0.012412        (45_6)
# 26  0.012272        (36_1)
# 44  0.011949         (8_5)
# 48  0.011233  (15_1, 33_4)
# 54  0.011191   (2_0, 38_4)
# 49  0.011177  (15_1, 38_4)
# 51  0.011120  (22_0, 25_0)
# 20  0.010840        (30_2)
# 55  0.010067   (38_4, 9_4)
# 50  0.010011  (15_1, 49_1)



# Şu anda elimizde olası Hizmet(ürün) veya Hizmet çiftleri ve bunlara karşılık support değerleri verilmiş.
# Burada 0.01'in altındaki olası değerler yok çünkü minimum support değerini(eşik değeri) 0.01 olarak vermiştik.
# Bunlar herbir hizmetin olasılığıdır. Bizim ihtiyacımız olan birliktelik kurallarıdır. Dolayısıyla bu veriyi kullanıp
# bunun üzerinden birliktelik kurallarını çıkaracağız.


# İhtiyacımız olan birliktelik kuralları için association_rules() metodu ile
# bu veriyi kullanıp bunun üzerinden birliktelik kurallarını çıkaracağız:

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

rules.head()
#   antecedents consequents  antecedent support  consequent support   support  confidence      lift  leverage  conviction  zhangs_metric
# 0     (13_11)       (2_0)            0.056627            0.130286  0.012819    0.226382  1.737574  0.005442    1.124216       0.449965
# 1       (2_0)     (13_11)            0.130286            0.056627  0.012819    0.098394  1.737574  0.005442    1.046325       0.488074
# 2      (15_1)       (2_0)            0.120963            0.130286  0.033951    0.280673  2.154278  0.018191    1.209066       0.609539
# 3       (2_0)      (15_1)            0.130286            0.120963  0.033951    0.260588  2.154278  0.018191    1.188833       0.616073
# 4      (15_1)      (33_4)            0.120963            0.027310  0.011233    0.092861  3.400299  0.007929    1.072262       0.803047

# antecedents: Önceki Hizmet
# consequents: İkinci Hizmet
# antecedent support: İlk Hizmetin tek başına gözlenme olasılığı
# consequent support: İkinci Hizmetin tek başına gözlenme olasılığı
# support: İki Hizmetin birlikte görülme olasılığı
# confidence: İlk hizmet alındığında  ikinci Hizmetin alınma olasılığı
# lift: Bir Hizmet alındığında ikinci Hizmetin alınma olasılığının kaç kat artacağının belirtir.
# leverage: lift benzeridir. Support u yüksek olan değerlere öncelik verme eğilimindedir bundan dolayı ufak bir yanlılığı vardır.
# conviction: Bir Hizmet olmadan diğer Hizmetin beklenen frakansı



#Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

# product_id: Öneri yapılmasını istediğimiz hizmetin id'si (stock_code olarak düşünebiliriz).
# rec_count: İstenen sayıya kadar tavsiye hizmeti getirir.
# İlk olarak en uyumlu ilk ürünü yakalayabilmek için kuralları lifte göre büyükten kücüğe sıraladık.
# (Bu sıralama tercihe göre confidence'e göre de olabilir.)
# Tavsiye edilecek ürünler için boş bir liste oluşturuyoruz.
# Sıralanmış kurallarda ilk önce gelen ürüne (hizmet) göre enumerate() metodunu kullanıyoruz.
# İkinci döngüde Hizmetlerde(product) gezilecek. Eğer tavsiye istenen hizmet yakalanırsa,
# index bilgisi i ile tutuluyordu bu index bilgisindeki consequents değerini recommendation_list'e ekle diyoruz.
# [0] ilk gördüğünü getirmesi için eklendi.

arl_recommender(rules,"2_0", 1)
# ['22_0']

arl_recommender(rules,"2_0", 3)
# ['22_0', '25_0', '15_1'] Hizmetleri önerilmiş oldu.

# Not: Önerilen hizmet sayısı arttıkça diğer denk gelen hizmetlerin(ürün) ilgili istatistiklerdeki değerleri daha düşük olacaktır.