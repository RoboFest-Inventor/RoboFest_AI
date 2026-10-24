#include <SoftwareSerial.h>
SoftwareSerial HC12(2, 3); // RX, TX

void setup() {
  Serial.begin(9600);
  HC12.begin(9600);
  Serial.println("--- 모터 원격 제어 송신기 ---");
  Serial.println("원하는 회전 수(숫자)를 입력하고 엔터를 누르세요. (예: 2, 5.5, -3)");
}

void loop() {
  // 시리얼 모니터에 입력한 값을 HC-12로 바이패스 송신
  if (Serial.available()) {
    delay(10);
    while (Serial.available()) {
      HC12.write(Serial.read());
    }
  }
  
  // 모터 아두이노로부터 피드백 메시지가 오면 화면에 출력
  if (HC12.available()) {
    while (HC12.available()) {
      Serial.print((char)HC12.read());
    }
  }
}