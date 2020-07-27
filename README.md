# rcv_badge_ssd

![image](https://user-images.githubusercontent.com/62923434/88531595-0ba2f480-d03e-11ea-8227-83e6f3d0cd41.png)

Sejong RCV Badge SSD project의 목표는 위와 같은 SSD논문 성능 원복입니다.
사용한 데이터 셋은 논문과 동일하게 pascal VOC dataset 2007 (trainval)(5011 images) 이고 , 평가는 pascal VOC dataset 2007 (test)입니다.

사용한 Data augmentation은 논문과 동일하게 진행했으며 300x300으로 resize하였습니다.
논문 뒷편에 나와 있는 new data augmentation인 zoom out은 포함하지 않았습니다.
이는 VOC 07 만 학습한 SSD300 논문 평가와 직접비교하기 위함입니다.

![image](https://user-images.githubusercontent.com/62923434/88532151-06927500-d03f-11ea-82c5-2264bc14b24b.png)

학습 시 learining rate설정은 논문에서와 같은 기준으로 적용했습니다.
![image](https://user-images.githubusercontent.com/62923434/88532824-3b52fc00-d040-11ea-9fef-fc09b30a0818.png)


train 코드로 학습한 클래스 별 최종 성능은 아래와 같습니다.

![image](https://user-images.githubusercontent.com/62923434/88532561-c8e21c00-d03f-11ea-98ee-d5d5ced1835f.png)
