% GENERATE WAVEFORM DATASET

function data = generate_waveform(N)

class_size=round(N/3);
u=rand(class_size,1);
for i=1:21
   x1(:,i)=u*waveform(i,1)+(1-u)*waveform(i,2)+randn(class_size,1);
end
x1(:,22)=1;
x1(:,23)=0;
x1(:,24)=0;

u=rand(class_size,1);
for i=1:21
   x2(:,i)=u*waveform(i,1)+(1-u)*waveform(i,3)+randn(class_size,1);
end
x2(:,22)=0;
x2(:,23)=1;
x2(:,24)=0;

u=rand(class_size,1);
for i=1:21
   x3(:,i)=u*waveform(i,2)+(1-u)*waveform(i,3)+randn(class_size,1);
end
x3(:,22)=0;
x3(:,23)=0;
x3(:,24)=1;

waveform1=[x1;x2;x3];

% shuffle the examples
sh=rand(3*class_size,1);

[thrash, q]=sort(sh);

data=waveform1(q,:);

return;