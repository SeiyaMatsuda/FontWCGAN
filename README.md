# FontWCGAN
##はじめに
本研究では条件付き敵対的生成ネットワーク（Conditional
Generative Adversarial Networks，GANs）[1] を
用いて，ユーザが指定した印象語を条件とした文字フォン
トを生成するモデルを構築する．文字フォントのデザイン
は専門のデザイナにより独占的に行われており，既存のソ
フトウェアにその創造性は備えられていない．そのため，
私たちはユーザが用途に応じて文字フォントを生成できる
ようなシステムを開発したいと考えた．近年のニューラル
ネットワークによる生成モデルの発展は，従来のデザイナ
による文字フォント生成という形からデザインの知識を持
たないユーザが用途に応じた文字フォントの生成を行うこ
とを可能にする．そのため，本研究では生成モデルの一つ
であるConditionalGAN を用いて，文字フォントを生成す
## 提案する生成モデル
図1 に提案する文字フォント生成ネットワークを示す．
通常のGAN では一様分布から画像生成を行うため, 生成
される画像を印象語によりコントロールすることができ
ない． そこで，GAN のネットワークに，クラスラベル
などを表す条件を与えることで，コントロールが可能な
モデルであるConditional GAN を用いて画像生成を行う．
また，学習を安定させるため，WGAN-gradient penalty
（WGAN-gp）［3］モデルで使用されている損失関数を用
いた．
## 使用する文字フォント・印象語
ネットワークの学習には，13 万以上の文字フォントを
提供しているWeb サイトであるMyFonts からMirza ら
の研究[2] により作成されたデータセットを使用する．各
文字フォントにはローマ字グリフ画像（26 文字の大文字
と小文字の両方を含む）と複数の印象語がタグラベルと
して与えられている．本研究では文字フォントのクラスを
“A” のみに絞り使用した．また，印象語をベクトル空間に
埋め込んだものを，条件として入力した．そのため，事前
学習済みの単語ベクトル化モデルの一つであるword2vec-
GoogleNews-vectors を使用してベクトル化する．なお文
字フォントに複数の印象語が付いている場合には，そのう
ち1 つをランダムに選んで条件として利用する．
## 実際に印象語を指定し生成された画像
 ### manuscriptを条件とし生成
![manuscript](./imgs/fake_manuscript.png)<br>
### shadingを条件とし生成
![shading](./imgs/fake_shading.png)<br>
### sansを条件とし生成
![sans](./imgs/fake_sans.png)<br>
### sansを条件とし生成
![decorative](./imgs/fake_decorative.png)<br>
## まとめ
特定印象を考慮した文字フォントの生成を行うことができるネットワークモデルを提案した．その結
果，一定の印象を反映したと考えられる文字フォントを生
成することに成功した．
## 参考文献
[1] M. Mirza, and S. Osindero, “Conditional generative
adver-sarial nets,” arXiv preprint arXiv1411.1784,
2014.<br>
[2] T. Chen, Z. Wang, N. Xu, H. Jin, and J.
Luo, “Large-scale Tag-based Font Retrieval with
Generative Feature Learning,” arXiv preprint
arXiv1909.02072, 2019.<br>
[3] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin,
and A.C. Courville, “Improved training of wasserstein
gans,” In Advances in Neural Information Processing
Systems, pp.5767-5777, 2017.<br>
