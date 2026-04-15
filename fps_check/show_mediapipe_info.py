#!/usr/bin/env python3
"""
MediaPipe Pose 상세 정보 확인 스크립트
노드 개수, 각 노드 이름, 연결 관계 등을 출력합니다.
"""
import mediapipe as mp

def show_mediapipe_info():
    """MediaPipe Pose의 상세 정보 출력"""
    
    mp_pose = mp.solutions.pose
    
    # Pose 모델 초기화 (model_complexity별로)
    model_configs = [
        (0, "Light (경량)"),
        (1, "Full (표준)"),
        (2, "Heavy (무거움)")
    ]
    
    for complexity, name in model_configs:
        print(f"\n{'='*70}")
        print(f"Model Complexity: {complexity} - {name}")
        print(f"{'='*70}")
        
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=complexity,
            smooth_landmarks=True
        )
        
        # 노드 정보
        print(f"\n📍 총 노드(키포인트) 개수: 33개")
        print(f"   각 노드는 (x, y, z, visibility) 정보를 포함")
        
        # 노드 이름 및 인덱스
        print(f"\n노드 상세 정보:")
        print(f"{'인덱스':<6} {'노드명':<30} {'설명':<40}")
        print("-" * 76)
        
        landmarks = [
            (0, "NOSE", "코"),
            (1, "LEFT_EYE_INNER", "왼쪽 눈 안쪽"),
            (2, "LEFT_EYE", "왼쪽 눈"),
            (3, "LEFT_EYE_OUTER", "왼쪽 눈 바깥쪽"),
            (4, "RIGHT_EYE_INNER", "오른쪽 눈 안쪽"),
            (5, "RIGHT_EYE", "오른쪽 눈"),
            (6, "RIGHT_EYE_OUTER", "오른쪽 눈 바깥쪽"),
            (7, "LEFT_EAR", "왼쪽 귀"),
            (8, "RIGHT_EAR", "오른쪽 귀"),
            (9, "MOUTH_LEFT", "입 왼쪽"),
            (10, "MOUTH_RIGHT", "입 오른쪽"),
            (11, "LEFT_SHOULDER", "왼쪽 어깨"),
            (12, "RIGHT_SHOULDER", "오른쪽 어깨"),
            (13, "LEFT_ELBOW", "왼쪽 팔꿈치"),
            (14, "RIGHT_ELBOW", "오른쪽 팔꿈치"),
            (15, "LEFT_WRIST", "왼쪽 손목"),
            (16, "RIGHT_WRIST", "오른쪽 손목"),
            (17, "LEFT_PINKY", "왼쪽 새끼손가락"),
            (18, "RIGHT_PINKY", "오른쪽 새끼손가락"),
            (19, "LEFT_INDEX", "왼쪽 검지"),
            (20, "RIGHT_INDEX", "오른쪽 검지"),
            (21, "LEFT_THUMB", "왼쪽 엄지손가락"),
            (22, "RIGHT_THUMB", "오른쪽 엄지손가락"),
            (23, "LEFT_HIP", "왼쪽 골반"),
            (24, "RIGHT_HIP", "오른쪽 골반"),
            (25, "LEFT_KNEE", "왼쪽 무릎"),
            (26, "RIGHT_KNEE", "오른쪽 무릎"),
            (27, "LEFT_ANKLE", "왼쪽 발목"),
            (28, "RIGHT_ANKLE", "오른쪽 발목"),
            (29, "LEFT_HEEL", "왼쪽 발뒤꿈치"),
            (30, "RIGHT_HEEL", "오른쪽 발뒤꿈치"),
            (31, "LEFT_FOOT_INDEX", "왼쪽 발가락"),
            (32, "RIGHT_FOOT_INDEX", "오른쪽 발가락"),
        ]
        
        for idx, name_en, name_ko in landmarks:
            print(f"{idx:<6} {name_en:<30} {name_ko:<40}")
        
        # 연결 관계 (skeleton)
        print(f"\n🔗 포즈 연결 관계 (Skeleton Connections):")
        
        connections = mp_pose.POSE_CONNECTIONS
        connection_list = list(connections)
        
        print(f"   총 연결 개수: {len(connection_list)}")
        print(f"\n   연결 목록:")
        for i, (start, end) in enumerate(connection_list, 1):
            start_name = landmarks[start][1]
            end_name = landmarks[end][1]
            print(f"   {i:2d}. {start} ({start_name}) ↔ {end} ({end_name})")
        
        # 모델 사양
        print(f"\n⚙️  모델 사양:")
        print(f"   Model Complexity: {complexity}")
        if complexity == 0:
            print(f"   입력 해상도: 256x256")
            print(f"   예상 FPS (라즈베리파이): 15-25")
            print(f"   메모리 사용: 적음")
        elif complexity == 1:
            print(f"   입력 해상도: 384x384")
            print(f"   예상 FPS (라즈베리파이): 8-15")
            print(f"   메모리 사용: 중간")
        else:
            print(f"   입력 해상도: 480x480")
            print(f"   예상 FPS (라즈베리파이): 3-8")
            print(f"   메모리 사용: 많음")
        
        pose.close()

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MediaPipe Pose - 상세 정보")
    print("=" * 70)
    
    try:
        show_mediapipe_info()
        
        print("\n" + "=" * 70)
        print("✅ Just Dance 게임 추천 설정")
        print("=" * 70)
        print("""
📌 추천 설정:
   - Model Complexity: 1 (Full) 또는 0 (Light)
   - Input resolution: 320x240 (웹캠 해상도)
   - 33개 노드 모두 사용 가능
   - 팔 동작: 노드 11~22 활용
   - 다리 동작: 노드 23~28 활용
   
👾 게임 구현 시:
   1. 타겟 포즈 저장 (학습 1): 예) "점프 준비" 포즈
   2. 실시간 포즈와 비교: 코사인 유사도 또는 각도 계산
   3. 임계값 설정: 일치도 > 70% = 정답
   4. 30 FPS 기준으로 30프레임(1초) 유지 시 +점수
        """)
        
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
