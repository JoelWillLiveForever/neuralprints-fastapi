import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import sys
import numpy as np
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(
        description="Инференс модели Keras по входным признакам.",
        add_help=False
    )
    parser.add_argument("features", type=float, nargs="*", help="Список признаков (например: 0.1 0.2 0.3 ...)")
    parser.add_argument("-I", "--interactive", action="store_true", help="Интерактивный режим (ввод с клавиатуры)")
    parser.add_argument("-V", "--verbose", action="store_true", help="Подробный вывод")
    parser.add_argument("-h", "--help", action="help", help="Показать это сообщение и выйти")

    args = parser.parse_args()

    expected_feature_count = {n_features}

    if args.interactive:
        print(f"Введите значения для {{expected_feature_count}} признаков через пробел:")
        try:
            features = list(map(float, input().strip().split()))
        except ValueError:
            print("Ошибка: Неверный формат ввода.")
            sys.exit(1)
    elif not args.features:
        parser.print_help()
        sys.exit(1)
    else:
        features = args.features

    if len(features) != expected_feature_count:
        print(f"Ошибка: ожидалось {{expected_feature_count}} признаков, получено {{len(features)}}.")
        sys.exit(1)

    if args.verbose:
        print("Загрузка модели и выполнение предсказания...")

    model = tf.keras.models.load_model("model.keras")
    input_array = np.array([features])
    prediction = model.predict(input_array)

    if args.verbose:
        print(f"Результат предсказания: {{prediction[0][0]}}")
    else:
        print(prediction[0][0])

if __name__ == "__main__":
    main()