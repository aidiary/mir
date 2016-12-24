#coding:utf-8
import os
import sys
import numpy as np
import numpy.linalg
import rpy2.robjects as robjects
from collections import defaultdict

# mir.py
# usage: python mir.py [sig file] [sig dir] [html file]
# sig file  : クエリ楽曲のシグネチャファイル
# sig dir   : 検索対象のシグネチャファイルのディレクトリ
# html file : 検索結果を出力するHTMLファイル

# 引数で指定したシグネチャファイルに近い
# 上位N件の楽曲を出力する

def KLDiv(mu1, S1, mu2, S2):
    """正規分布間のカルバック・ライブラー情報量"""
    # 逆行列を計算
    try:
        invS1 = np.linalg.inv(S1)
    except numpy.linalg.linalg.LinAlgError:
        raise;
    try:
        invS2 = np.linalg.inv(S2)
    except numpy.linalg.linalg.LinAlgError:
        raise;

    # KL Divergenceを計算
    t1 = np.sum(np.diag(np.dot(invS2, S1)))
    t2 = (mu2 - mu1).transpose()
    t3 = mu2 - mu1
    return t1 + np.dot(np.dot(t2, invS2), t3)

def symKLDiv(mu1, S1, mu2, S2):
    """対称性のあるカルバック・ライブラー情報量"""
    return 0.5 * (KLDiv(mu1, S1, mu2, S2) + KLDiv(mu2, S2, mu1, S1))

# Rで輸送問題を解くライブラリ
# Rのデフォルトパッケージではないのでインストールが必要
# Rでinstall.packages("lpSolve")
robjects.r['library']('lpSolve')
transport = robjects.r['lp.transport']

def calcEMD(sigFile1, sigFile2):
    # シグネチャをロード
    sig1 = loadSignature(sigFile1)
    sig2 = loadSignature(sigFile2)

    # 距離行列を計算
    numFeatures = sig1.shape[0]                 # クラスタの数
    dist = np.zeros(numFeatures * numFeatures)  # 距離行列（フラット形式）

    for i in range(numFeatures):
        mu1 = sig1[i, 1:21].reshape(20, 1)   # 縦ベクトル
        S1 = sig1[i, 21:421].reshape(20, 20)
        for j in range(numFeatures):
            mu2 = sig2[j, 1:21].reshape(20, 1)
            S2 = sig2[j, 21:421].reshape(20, 20)
            # 特徴量iと特徴量j間のKLダイバージェンスを計算
            dist[i * numFeatures + j] = symKLDiv(mu1, S1, mu2, S2)

    # シグネチャの重み（0列目）を取得
    w1 = sig1[:,0]
    w2 = sig2[:,0]

    # 重みと距離行列からEMDを計算
    # transport()の引数を用意
    costs = robjects.r['matrix'](robjects.FloatVector(dist),
                                 nrow=len(w1), ncol=len(w2),
                                 byrow=True)
    row_signs = ["<"] * len(w1)
    row_rhs = robjects.FloatVector(w1)
    col_signs = [">"] * len(w2)
    col_rhs = robjects.FloatVector(w2)

    t = transport(costs, "min", row_signs, row_rhs, col_signs, col_rhs)
    flow = t.rx2('solution')

    dist = dist.reshape(len(w1), len(w2))
    flow = np.array(flow)
    work = np.sum(flow * dist)
    emd = work / np.sum(flow)
    return emd

def loadSignature(sigFile):
    """シグネチャファイルをロード"""
    mat = []
    fp = open(sigFile, "r")
    for line in fp:
        line = line.rstrip()
        mat.append([float(x) for x in line.split()])
    fp.close()
    return np.array(mat)

def getArtist(mp3Path):
    """MP3ファイルからアーティストを取得"""
    import eyeD3
    try:
        tag = eyeD3.Tag()
        tag.link(mp3Path)
        artist = tag.getArtist()
    except:
        artist = "None"
    # 空白のとき
    if artist == "": artist = "None"
    return artist

def makeHTML(ranking, htmlFile, N=10):
    """ランキングをHTML形式で出力"""
    import codecs
    fout = codecs.open(htmlFile, "w", "utf-8")

    # HTMLヘッダを出力
    fout.write('<!DOCTYPE html>\n')
    fout.write('<html lang="ja">\n')
    fout.write('<head><meta charset="UTF-8" /><title>%s</title></head>\n' % htmlFile)
    fout.write('<body>\n')
    fout.write('<table border="1">\n')
    fout.write(u'<thead><tr><th>ランク</th><th>EMD</th><th>タイトル</th>')
    fout.write(u'<th>アーティスト</th><th>音声</th></tr></thead>\n')
    fout.write(u'<tbody>\n')

    # ランキングを出力
    rank = 1
    for sigFile, emd in sorted(ranking.items(), key=lambda x:x[1], reverse=False)[:N]:
        prefix = sigFile.replace(".sig", "")

        # rawをwavに変換（HTMLプレーヤー用）
        rawPath = os.path.join("raw", prefix + ".raw")
        wavPath = os.path.join("wav", prefix + ".wav")
        if not os.path.exists("wav"): os.mkdir("wav")
        os.system('sox -r 16000 -e signed-integer -b 16 "%s" "%s"' % (rawPath, wavPath))

        # アーティスト名を取得
        mp3Path = os.path.join("mp3", prefix + ".mp3")
        artist = getArtist(mp3Path)

        # HTML出力
        # HTML5のオーディオプレーヤーを埋め込む
        audio = '<audio src="%s" controls>' % wavPath
        fout.write("<tr><td>%d</td><td>%.2f</td><td>%s</td><td>%s</td><td>%s</td></tr>\n"
                   % (rank, emd, prefix, artist, audio))
        rank += 1

    fout.write("</tbody>\n");
    fout.write("</table>\n")
    fout.write("</body>\n")
    fout.write("</html>\n")
    fout.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "python mir.py [sig file] [sig dir] [html file]"
        sys.exit()

    targetSigPath = sys.argv[1]
    sigDir = sys.argv[2]
    htmlFile = sys.argv[3]

    ranking = defaultdict(float)

    # 全楽曲との間で距離を求める
    for sigFile in os.listdir(sigDir):
        sigPath = os.path.join(sigDir, sigFile)
        emd = calcEMD(targetSigPath, sigPath)
        if emd < 0: continue
        ranking[sigFile] = emd

    # ランキングをEMDの降順にソートして出力
    N = 10
    rank = 1
    for sigFile, emd in sorted(ranking.items(), key=lambda x:x[1], reverse=False)[:N]:
        print "%d\t%.2f\t%s" % (rank, emd, sigFile)
        rank += 1

    # EMDの昇順に上位10件をHTMLにして出力
    makeHTML(ranking, htmlFile, N)
