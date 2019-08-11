import pandas as pd
from kaggle_sample.Iris.my_decision_tree_01.DecisionTree import DecisionTree
from typing import List

TRAIN_TSV_PATH: str = r'../data/train.tsv'
TEST_TSV_PATH: str = r'../data/test.tsv'
OUTPUT_PATH: str = r'../data/output.csv'


def main():
    # train.tsvをpandasで読み込み
    train_df = pd.read_csv(TRAIN_TSV_PATH, delimiter='\t')
    print(train_df.head())

    # 学習用の説明変数をnumpyで読み込む
    train_np = train_df[['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm']].values
    print(train_np)
    print(type(train_np))

    # 決定木のclassをnumpyで取得
    train_class_np = train_df['class'].values
    print(train_class_np)
    print(type(train_class_np))

    # 決定木を作成、学習を行う
    decision_tree: DecisionTree = DecisionTree(criterion=0.005)
    decision_tree.fit(train_np, train_class_np)
    # 木出力
    decision_tree.print_tree()

    # 試験データを読み込みテストを実施
    test_df = pd.read_csv(TEST_TSV_PATH, delimiter='\t')
    test_np = test_df[['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm']].values
    answer_np = decision_tree.predict(test_np)
    print(answer_np)

    # csv出力
    output_list: List[str] = []
    for i in range(len(answer_np)):
        output_list.append(str(test_df["id"][i]) + "," + answer_np[i])

    print(output_list)
    with open(OUTPUT_PATH, mode='w') as writer:
        writer.write('\n'.join(output_list))


if __name__ == '__main__':
    main()
