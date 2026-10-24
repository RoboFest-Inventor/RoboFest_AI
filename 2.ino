#include <SoftwareSerial.h>

SoftwareSerial HC12(9, 10); // RX, TX

// 엔코더 핀
const byte ENC_A = 2;
const byte ENC_B = 3;
const float PULSES_PER_REV = 1920.0; // 출력축 1회전당 펄스

// 모터 드라이버 제어 핀 (환경에 맞게 수정 가능)
const byte MOTOR_PWM = 6;  
const byte MOTOR_IN1 = 11;  
const byte MOTOR_IN2 = 12;  

volatile long encoderCount = 0;

// 모터 제어 변수
long targetPulses = 0;
bool isMoving = false;
int motorSpeed = 150; // 모터 구동 속도 (0 ~ 255)

void onEncoderA() {
  if (digitalRead(ENC_B) == LOW) {
    encoderCount++;
  } else {
    encoderCount--;
  }
}

void moveMotor(int dir, int speed) {
  analogWrite(MOTOR_PWM, speed);
  if (dir == 1) { 
    digitalWrite(MOTOR_IN1, HIGH);
    digitalWrite(MOTOR_IN2, LOW);
  } else if (dir == -1) { 
    digitalWrite(MOTOR_IN1, LOW);
    digitalWrite(MOTOR_IN2, HIGH);
  } else { 
    analogWrite(MOTOR_PWM, 0);
    digitalWrite(MOTOR_IN1, LOW);
    digitalWrite(MOTOR_IN2, LOW);
  }
}

void setup() {
  Serial.begin(9600);
  HC12.begin(9600);

  pinMode(ENC_A, INPUT_PULLUP);
  pinMode(ENC_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_A), onEncoderA, RISING);

  pinMode(MOTOR_PWM, OUTPUT);
  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);
  moveMotor(0, 0); 

  Serial.println("모터 제어 수신기 준비 완료");
}

void loop() {
  // 1. 무선(HC-12)으로 목표 바퀴 수 데이터 수신 프로세스
  if (HC12.available() && !isMoving) {
    float targetRevolutions = HC12.parseFloat(); 
    
    if (targetRevolutions != 0.0) {
      // 새로운 명령 시작 시 엔코더 카운트 0으로 리셋
      noInterrupts();
      encoderCount = 0;
      interrupts();

      // 절대값 기준으로 목표 펄스 수 계산
      targetPulses = abs(targetRevolutions * PULSES_PER_REV);
      isMoving = true;

      HC12.print("-> 명령 접수: ");
      HC12.print(targetRevolutions, 2);
      HC12.print("바퀴 이동 시작 (목표 펄스: ");
      HC12.print(targetPulses);
      HC12.println(")");

      // 모터 출발 방향 설정
      if (targetRevolutions > 0) {
        moveMotor(1, motorSpeed);  
      } else {
        moveMotor(-1, motorSpeed); 
      }
    }
  }

  // 2. 모터가 회전 중일 때 엔코더 체크 및 정지 프로세스
  if (isMoving) {
    noInterrupts();
    long currentCount = encoderCount; // 현재까지 기록된 실제 펄스값 복사
    interrupts();

    // 목표 펄스값에 도달했는지 확인 (방향 무관하게 절대값으로 비교)
    if (abs(currentCount) >= targetPulses) {
      moveMotor(0, 0); // 모터 정지
      isMoving = false;

      // [수정] 무선 수신측 시리얼 모니터로 최종 엔코더 값 전송
      HC12.print("-> [이동 완료] 최종 엔코더 값: ");
      HC12.print(currentCount); // 정회전이면 양수, 역회전이면 음수로 출력됨
      HC12.print(" 펄스 (약 ");
      HC12.print((float)currentCount / PULSES_PER_REV, 2);
      HC12.println(" 바퀴 이동됨)");
    }
  }
}
 