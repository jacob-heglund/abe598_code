close all
clear all

load data_BKR
figure(1)
errorbar(t,rew_total,std_total,'g');
hold on

clear all
load data_GP%data_approx_20features
figure(1)
errorbar(t,rew_total,std_total,'c:');

clear all
load data_BKR_CL
figure(1)
errorbar(t,rew_total,std_total,'-.');

clear all
load data_noapprox
figure(1)
errorbar(t,rew_total,std_total,'r--');
xlabel('episodes')
ylabel('Reward')
%legend('with BKR-Q learning','th approx<Nstate','tabular',0)
legend('with BKR-Q learning','with GP','BKR-CL','tabular',0)

% 
% set(figure(2),'Position',[100 100 800 550]);
% set(figure(2),'PaperOrientation','portrait','PaperSize',[8.5 6.0],'PaperPositionMode', 'auto', 'PaperType','<custom>');
% saveas(figure(2),'phage_bact_sim_controlled','pdf')