Bu ödev 21120606021 numaralı Cemre Sırma Tarafından BIL485 dersi kapsamında hazırlanmış olan fotoğraflardan kıyafet türünü tespit etme modeli oluşturma ödevidir.

Bu repoda, MNIST veri setiyle eğitilmiş CNN modelinin kodları ve telefonumun kamerasıyla çekmiş olduğum bazı kıyafetlerimin görselleri ve bunlara dair test görselleri yer almaktadır.

Deney Sonucu Hakkında Genel Değerlendirme
Tarafınıza sunulmış olan bu el yazısı tanıma modeli tıpkı ödev isterlerinde belirtmiş olduğunuz gibi CNN mimarisi kullanılmak suretiyle hazırlanmıştır.
Geliştirilen modeli kullanmak suretiyle tarafımca yapılan testlerde MNIST test seti üzerinde yaklaşık %91.66 doğruluk (accuracy) ve 0.10 kayıp (loss) değerleri elde edilmiştir.
Söz konusu bu modelin gerçek hayat koşullarında hakikaten nasıl bir davranış sergileyeceğini test etmek üzere sizin de ödev isterlerinde belirtmiş olduğunuz gibi kıyafet dolabımda bulunan rastgele 5 kategoriden toplamda 12 tane kıyafet ve 3 tane ayakkabının fotoğrafını telefonumdan çektim.
Ki yine ödev isterlerinde belirtmiş olduğunuz gibi bu taramanın görselleri bu repo içerisinde tarafınıza sunulmuştur.

| Kategori        | Örnek Sayısı | Öne Çıkan Top-1 Güven Skorları | Çıkarım Süresi Aralığı |
| `T-shirt/top`   |       3      | Yanlış Sınıflandırıldı         | — |
| `Pantolon`      |       3      | %99, %81, %77                  | — |
| `Ceket`         |       3      | %92, %63, %50                  | — |
| `Çanta`         |       3      | %100, %91, %61                 | — |
| `Spor Ayakkabı` |       3      | %92, %84, %74                  | — |


| Alan                  | Açıklama |
| Project Title         | MNIST kullanılarak CNN temelli kıyafet çeşitlerine dair görüntü sınıflandırma |
| Problem               | 10 farklı kıyafet kategorisini (T-shirt, Pantolon, Kazak, Elbise, Ceket, Sandaler, Gömlek, Spor Ayakkabı, Çanta, Yarım Bot) görüntülerden sınıflandırmaya yönelik çok sınıflı görüntü sınıflandırma problemi |
| Dataset               | fashion-mnist |
| Dataset URL           | https://github.com/zalandoresearch/fashion-mnist.git |
| Model Architecture    | CNN Convolution Neural Network |
| Input Size            | 28 * 28 * 1 |
| Hyperparameters       | Learning Rate: 0.001 , Batch Size: 64 , Epochlar: 10 |
| Optimizer             | Adaptive Moment Estimation |
| Activation            | Gizli katmanlarda ReLU, 10 sınıflı çıkış katmanında olasılık dağılımı üretmek için Softmax fonksiyonları kullanılmıştır|
| LossFunction          | Sparse Categorical Crossentropy ile Categorical Crossentropy fonksiyonları kullanılmıştır |
| Evaluation Metrics    | Accuracy: 0.9166 , Precision: 0.9193 , Recall: 0.9166 , F1 Score: 0.9171 |
| Training Environment  | PyCharm Community Edition 2024.1 |
| Best Performance      | Test veri setinde %91.63 Test Doğruluğu. Gerçek dünya fotoğraflarında uygun açı ve kontrast sağlandığında 5 kategoride 12/15 (%80 Top-1) sınıflandırma başarısı | 
| Inference Time        | 270.49 ms – 473.82 ms aralığında |
| Deployment Technology | Gradio web arayüzünü Python, TensorFlow/Keras ve OpenCV/Pillow ile birlikte kullanarak oluşturdum |
| Repository            | https://github.com/21120606021-Cemre-Sirma/21120606021-Cemre-Sirma-MNIST-Kiyafet-Tanima-CNN.git |