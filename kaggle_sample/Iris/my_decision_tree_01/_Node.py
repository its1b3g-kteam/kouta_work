import numpy as np
from typing import Dict


class _Node:
    def __init__(self):
        """
        初期処理
        left       : 左の子ノード（しきい値未満）
        right      : 右の子ノード（しきい値以上）
        feature    : 分割する特徴番号
        threshold  : 分割するしきい値
        label      : 割り当てられたクラス番号
        numdata    : 割り当てられたデータ数
        gini_index : 分割指数（Giniインデックス）
        """
        self.left: _Node = None
        self.right: _Node = None
        self.feature: int = None
        self.threshold = None
        self.label = None
        self.numdata: int = None
        self.gini_index = None

    def build(self, data, target):
        """

        :param data: データ numpy
        :param target: 分類クラス numpy
        :return:
        """

        # arr_np = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
        # arr_np.shape[0] -> 3
        # arr_np.shape[1] -> 5
        # arr_np[:,0] -> [1,6,11]
        # arr_np[0,:] -> [1,2,3,4,5]

        self.numdata = data.shape[0]
        num_features: int = data.shape[1]

        # 全データが同一クラスとなったら分割終了
        if len(np.unique(target)) == 1:
            self.label = target[0]
            return

        # 自分のクラスを設定(各データの多数決)
        # Dict型でtarget の各値の個数をカウントする  target=[1,0,1,1] -> class_cnt={0:1, 1:3}
        class_cnt: Dict[int:int] = {i: len(target[target == i]) for i in np.unique(target)}
        # カウント値が最大のtupleを取り出す max_item=(1, 3)
        max_item: tuple = max(class_cnt.items(), key=lambda x: x[1])
        self.label = max_item[0]

        # 最良の分割を記憶する変数
        # 不純度変化なし
        best_gini_index = 0.0
        best_feature = None
        best_threshold = None

        # 自分の不純度は先に計算しておく
        gini = self.gini_func(target)

        # 項目ごとにfor文を実施
        for f in range(num_features):
            # 分割候補の計算

            # f番目の特徴量（重複排除） arr_np_0 = arr_np[:, 0]  -> [1,6,11]
            data_f = np.unique(data[:, f])
            # 前後の中間値を計算 arr_np_0[:-1] = [1,6]   arr_np_0[1:] = [6,11]  (arr_np_0[:-1] + arr_np_0[1:])/2 = [3.5,8.5]
            points = (data_f[:-1] + data_f[1:]) / 2.0

            # 分割を試す
            for threshold in points:

                # しきい値で2グループに分割
                target_l = target[data[:, f] < threshold]
                target_r = target[data[:, f] >= threshold]

                # 分割後の不純度からGiniインデックスを計算
                gini_l = self.gini_func(target_l)
                gini_r = self.gini_func(target_r)
                pl = float(target_l.shape[0]) / self.numdata
                pr = float(target_r.shape[0]) / self.numdata
                # 分割前のgini係数から分割後のgini係数の値の差を算出する
                gini_index = gini - (pl * gini_l + pr * gini_r)

                # 最良の分割結果を保存
                if gini_index > best_gini_index:
                    best_gini_index = gini_index
                    best_feature = f
                    best_threshold = threshold

        # 不純度が減らなければ終了
        if best_gini_index == 0:
            return

        # 最良の分割を保持する
        self.feature = best_feature
        self.gini_index = best_gini_index
        self.threshold = best_threshold

        # 左右の子を作って再帰的に分割させる
        data_l = data[data[:, self.feature] < self.threshold]
        target_l = target[data[:, self.feature] < self.threshold]
        self.left = _Node()
        self.left.build(data_l, target_l)

        data_r = data[data[:, self.feature] >= self.threshold]
        target_r = target[data[:, self.feature] >= self.threshold]
        self.right = _Node()
        self.right.build(data_r, target_r)

    def gini_func(self, target) -> float:
        """Gini関数の計算"""
        # https://www.randpy.tokyo/entry/decision_tree_theory
        classes = np.unique(target)
        numdata = target.shape[0]

        # Gini関数本体
        gini: float = 1.0
        for c in classes:
            gini -= (len(target[target == c]) / numdata) ** 2.0

        return gini

    def prune(self, criterion, numall):
        """木の剪定を行う
        criterion  : 剪定条件（この数以下は剪定対象）
        numall    : 全ノード数
        """
        # https://www.randpy.tokyo/entry/decision_tree_theory_pruning
        # 自分が葉ノードであれば終了
        if self.feature is None:
            return

        # 子ノードの剪定
        self.left.prune(criterion, numall)
        self.right.prune(criterion, numall)

        # 子ノードが両方葉であれば剪定チェック
        if self.left.feature is None and self.right.feature is None:

            # 分割の貢献度：GiniIndex * (データ数の割合)
            result = self.gini_index * float(self.numdata) / numall

            # 貢献度が条件に満たなければ剪定する
            if result < criterion:
                self.feature = None
                self.left = None
                self.right = None

    def predict(self, d):
        """ 入力データ（単一）の分類先クラスを返す """

        # 節の場合、子ノードのクラスを見に行く
        if self.feature is not None:
            if d[self.feature] < self.threshold:
                return self.left.predict(d)
            else:
                return self.right.predict(d)

        # 葉ノードの場合、分類クラス
        else:
            return self.label

    def print_tree(self, depth: int, tf: str):
        """ 出力 """

        header: str = "\t" * depth + tf + " ->"

        if self.feature is not None:
            print(header + str(self.feature) + " < " + str(self.threshold) + "?")
            self.left.print_tree(depth + 1, "T")
            self.right.print_tree(depth + 1, "F")

        else:
            print(header + "{" + str(self.label) + " : " + str(self.numdata) + "}")
