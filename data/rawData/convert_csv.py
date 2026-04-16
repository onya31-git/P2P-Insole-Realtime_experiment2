import os
def fix_semicolon_csv(input_filename, output_filename):
    """
    セミコロン区切りのファイルをカンマ区切りに変換する関数
    """
    try:
        # 1. 元のファイルを読み込む
        with open(input_filename, 'r', encoding='utf-8') as file:
            content = file.read()

        # 2. セミコロンをカンマに置換する
        fixed_content = content.replace(';', ',')

        # 3. 新しいCSVファイルとして書き出す
        with open(output_filename, 'w', encoding='utf-8', newline='') as file:
            file.write(fixed_content)
        
        print(f"変換が完了しました！: {output_filename}")

    except FileNotFoundError:
        print(f"エラー: {input_filename} が見つかりませんでした。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

# --- 設定項目 ---
# 読み込むファイル名（ここに元のファイル名を入れてください）
input_file = 'skeleton.csv' 
# 保存するファイル名
output_file = 'fixed_skeleton.csv'

# 実行
if __name__ == "__main__":
    fix_semicolon_csv(input_file, output_file)