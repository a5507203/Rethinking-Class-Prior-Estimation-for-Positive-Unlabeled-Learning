% waveform.m

% makes responses for three different classes of waveforms

function y=waveform(t,C)

if C==1
   if t>1 & t<=7
      y=t-1;
   elseif t>7 & t<13
      y=(7-t)+6;
   else
      y=0;
   end
elseif C==2
   if t>9 & t<15
      y=t-9;
   elseif t>=15 & t<21
      y=(15-t)+6;
   else
      y=0;
   end
else
   if t>5 & t<11
      y=t-5;
   elseif t>=11 & t<17
      y=(11-t)+6;
   else
      y=0;
   end
end

return