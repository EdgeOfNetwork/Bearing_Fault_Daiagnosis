# Bearing_Fault_Daiagnosis
  
한번에 다 코딩하려 하지 말고  

계획을 세워서  
꾸준히 커밋합시다.  

========================2/20 - 23 
WIP :  
[-] pymSSA 관련 코드 숙지 및 bearing data mSSA에 적용해보기  
:데이터 사이즈가 너무 커서 램 용량이 터진다. 패키지 사용 보류   
  
[V] Transformer encoder파트만 짜른거 1D 데이터 돌리보기 -> finetune까지 해보기 
:대강 95%나온다.   
  
[V] bearing data에 노이즈 넣어보기(백색 가우시안 노이즈)    
[V] bearing data를 univariate하게 변형 후 SSA 적용해보기  
[V] SSA 적용된 데이터 시각화 해보기  
[V] denoising 여부 확인해보기  
    

[] 1.실험용 데이터셋 최종 결정    
[] 2.데이터셋 + noise 버전 제작  
[] 3.noise 버전 -> SSA denoisee 버전 데이터셋  

[] 비교모델 선정 및 코드확보  

[] 논문 참고 및 main proposal 용 TR모델 개선  


[] Wandb로 fine-tune 해보기   
:우선순위 뒤로 넘기자  
[] encoder를 위한 모듈화 해보기  
  
=========================
[V] 페이퍼 SSA관련 내용 작성  (2절)  
[] SSA관련 내용 축소 및 3절에 분해능력 관련 작성 (2~3절)  
[] 트랜스포머 related work 작성  (2절)    

[] mSSA로 multivariate 내용 3절 작성   
