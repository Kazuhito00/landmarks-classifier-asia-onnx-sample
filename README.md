# landmarks-classifier-asia-onnx-sample
<img src="https://user-images.githubusercontent.com/37477845/150822765-21b5d72d-3053-49fa-af43-dd7dd109aeef.gif" width="45%"><br>
[landmarks_classifier_asia_V1](https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1)のPythonでのONNX推論サンプルです。<br>
ONNXに変換したモデルも同梱しています。<br>
変換自体を試したい方はColaboratoryなどで[landmarks_classifier_asia_v1_sample.ipynb](landmarks_classifier_asia_v1_sample.ipynb)を使用ください。<br>

# Requirement(ONNX推論)
* OpenCV 4.5.3.56 or later
* onnxruntime 1.9.0 or later 

# Demo
デモの実行方法は以下です。
```bash
python sample_onnx.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：onnx/landmarks_classifier_asia_v1.onnx
* --input_size<br>
モデルの入力サイズ<br>
デフォルト：321,321

# Reference
* [landmarks_classifier_asia_V1](https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
landmarks-classifier-asia-onnx-sample is under [Apache-2.0 License](LICENSE).

# License(Image)
雷門の画像は[フリー素材ぱくたそ](https://www.pakutaso.com)様の写真を利用しています。
