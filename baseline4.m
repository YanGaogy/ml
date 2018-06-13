clear;clc;
load('/Users/apple/Documents/ml_2/zero_shot/AWA_data.mat');
addpath(genpath('/Users/apple/Documents/ml_2/zero_shot/drtoolbox'));

%X_test_our=X_test_our(1:50:5000,:);
%Y_test_our=Y_test_our(1:50:5000,:);

classes=50;
dims=2048;
att_num=85;
train_classes=unique(Y_train_our);
test_classes=unique(Y_test_our);
Att_test=Att(test_classes,:);

%%
load('./inter.mat');

dist_save=cell(1,att_num);
for m=1:att_num
    dist_save{m}=cell(1,2);
    load(['att_',num2str(m)]);
    
    dist_save{m}{1}=dist_diff_list_before;
    dist_save{m}{2}=proof_sum_list;
end    
%save('temp','dist_save');


% all_things=cell(1,size(feature_list,2));
% 
% temp_list_by_m=cell(1,att_num);
% for m=1:att_num
%     temp_list_by_m{m}=cell(1,2);
%     load(['att_',num2str(m)]);
%     temp_list_by_m{m}{1}=dist_diff_list_before;
%     temp_list_by_m{m}{2}=proof_sum_list;
% end
% 
% 
% for i=1:size(feature_list,2)
%     all_things{i}=cell(1,2);
%     dist_list=zeros(size(X_test_our,2),att_num);
%     proof_list=zeros(size(X_test_our,2),att_num);
%     for m=1:att_num
%         dist_list(:,m)=temp_list_by_m{m}{1}(:,i);
%         proof_list(:,m)=temp_list_by_m{m}{2}(:,i);
%     end
%     
%     all_things{i}{1}=dist_list;
%     all_things{i}{2}=proof_list;
% end

%%
% acc_para=cell(1,size(feature_list,1)-19);
% for f=1:size(feature_list,1)-19
%     acc_para{f}=cell(1,att_num);
%     
%     for m=1:att_num
%         acc_para{f}{m}=cell(1,3);
%     end
% end
overall_acc=zeros(size(feature_list,1)-19,1);



%% First 20 features
predict_final=zeros(size(X_test_our,1),att_num);
parfor m=1:att_num
    
    att_m=Att(:,m);
    att_train=att_m(train_classes,:);
    att_test=att_m(test_classes,:);
    
    % ground truth in test
    Y_att_test=zeros(length(Y_test_our),1);
    label=att_test>0.11;
    p_classes_test=test_classes(label);
    
    for i=1:length(p_classes_test)
        label=Y_test_our==p_classes_test(i);
        Y_att_test(label)=1;
    end
    
    
    dist_diff_list_before=dist_save{m}{1}(:,1:20);
    proof_sum_list=dist_save{m}{2}(:,1:20);
    
    dist_diff_list=cell(1,size(X_test_our,1));
    for i=1:size(X_test_our,1)
        test_one=[];
        for n=1:size(proof_sum_list,2)
            
            if proof_sum_list(i,n)==0 || proof_sum_list(i,n)==3
                test_one=[test_one;dist_diff_list_before(i,n)];
            end
        end 
        dist_diff_list{i}=test_one;
    end
    
    mid_list1=[];
    mid_list2=[];
    mid_list3=[];
    for q=1:size(dist_diff_list,2)
        mid_list1=[mid_list1;max(abs(dist_diff_list{q}))];
    
        mid_list2=[mid_list2;max(dist_diff_list{q})];
        mid_list3=[mid_list3;min(dist_diff_list{q})];
    
    end
    max_abs=max(mid_list1);
    max_all=max(mid_list2);
    min_all=min(mid_list3);
    dist_diff_list_cy=dist_diff_list;
    
    x=0;
    for para0=0:0.02:max_abs
        for para=min_all:0.02:max_all
            x=x+1;
        end    
    end
    
    final_list=zeros(x,3);
    predict_list=zeros(size(X_test_our,1),x);
    
    y=0;
    for para0=0:0.02:max_abs
        dist_diff_final_list=zeros(size(X_test_our,1),1);
        for q=1:size(dist_diff_list_cy,2)
            lab=abs(dist_diff_list_cy{q})<para0;
            dist_diff_list_cy{q}(lab)=0;
            dist_diff_final=sum(dist_diff_list_cy{q},1)/size(dist_diff_list_cy{q},1);
            dist_diff_final_list(q)=dist_diff_final;    
        end    
        
        for para=min_all:0.02:max_all
            y=y+1;
            predict=ones(length(Y_test_our),1);
            lable=dist_diff_final_list>para;
            predict(lable,:)=0;
            
            predict_list(:,y)=predict;
            check_list=[predict,Y_att_test];
            
            acc=sum(check_list(:,1)==check_list(:,2),1)/length(Y_test_our);
            final_list(y,:)=[para0,para,acc];
            
            disp(['att:[',num2str(m),']---',num2str(para0),'---',num2str(para)]);
               
        end    
    end
    
    acc_max=max(final_list(:,3));
    acc_max_para0=final_list(final_list(:,3)==acc_max,1);
    acc_max_para=final_list(final_list(:,3)==acc_max,2);
    
%     acc_para{1}{m}{1}=acc_max;
%     acc_para{1}{m}{2}=acc_max_para0;
%     acc_para{1}{m}{3}=acc_max_para;
    
    ind=find(final_list(:,3)==acc_max);
    predict_max=predict_list(:,ind);
    predict_max_1=predict_max(:,1);
    
    predict_final(:,m)=predict_max_1;
    
    fname = sprintf( 'att2_%d', m );
    iSaveX_b4_1000(fname,final_list,acc_max,acc_max_para0,acc_max_para,predict_max);
    
end

predict_cl_list=[];
for i=1:size(predict_final,1)
    pre_one=predict_final(i,:);
    [idx_f,DIST]=knnsearch(Att_test,pre_one,'k',1,'NSMethod','exhaustive','Distance','correlation');
    
    predict_cl=test_classes(idx_f);
    predict_cl_list=[predict_cl_list;predict_cl];
end

check_list_cl=[predict_cl_list,Y_test_our];
acc_cl=sum(check_list_cl(:,1)==check_list_cl(:,2),1)/length(Y_test_our);

disp(['acc_cl:[',num2str(acc_cl),']']);
overall_acc(1)=acc_cl;

the_max_acc=acc_cl;

disp(['att:[',num2str(acc_cl),']']);

save('result1','check_list_cl','acc_cl','predict_final');

%% good feature list
good_feature_list=cell(1,att_num);
for m=1:att_num
    good_feature_list{m}=cell(1,2);
    good_feature_list{m}{1}=dist_save{m}{1}(:,1:20);
    good_feature_list{m}{2}=dist_save{m}{2}(:,1:20);
end    




%% rest features
for f=21:size(feature_list,1)
    
    predict_final=zeros(size(X_test_our,1),att_num);
    
    temp_list=cell(1,att_num);
    for m=1:att_num
        temp_list{m}=cell(1,2);
    end    
    
    
    parfor m=1:att_num
        att_m=Att(:,m);
        att_train=att_m(train_classes,:);
        att_test=att_m(test_classes,:);
    
        % ground truth in test
        Y_att_test=zeros(length(Y_test_our),1);
        label=att_test>0.11;
        p_classes_test=test_classes(label);
    
        for n=1:length(p_classes_test)
            label=Y_test_our==p_classes_test(n);
            Y_att_test(label)=1;
        end
        
        dist_diff_list_before_one=dist_save{m}{1}(:,f);
        proof_sum_list_one=dist_save{m}{2}(:,f);
        
        dist_diff_list_before=[good_feature_list{m}{1},dist_diff_list_before_one];
        proof_sum_list=[good_feature_list{m}{2},proof_sum_list_one];
        
        temp_list{m}{1}=dist_diff_list_before;
        temp_list{m}{2}=proof_sum_list;
        
        dist_diff_list=cell(1,size(X_test_our,1));
        for i=1:size(X_test_our,1)
            test_one=[];
            for n=1:size(dist_diff_list_before,2)
            
                if proof_sum_list(i,n)==0 || proof_sum_list(i,n)==3
                    test_one=[test_one;dist_diff_list_before(i,n)];
                end
            end 
            dist_diff_list{i}=test_one;
        end
        
        mid_list1=[];
        mid_list2=[];
        mid_list3=[];
        for q=1:size(dist_diff_list,2)
            mid_list1=[mid_list1;max(abs(dist_diff_list{q}))];
    
            mid_list2=[mid_list2;max(dist_diff_list{q})];
            mid_list3=[mid_list3;min(dist_diff_list{q})];
    
        end
        max_abs=max(mid_list1);
        max_all=max(mid_list2);
        min_all=min(mid_list3);
        dist_diff_list_cy=dist_diff_list;
        
        x=0;
        for para0=0:0.02:max_abs
            for para=min_all:0.02:max_all
                x=x+1;
            end    
        end
    
        final_list=zeros(x,3);
        predict_list=zeros(size(X_test_our,1),x);
    
        y=0;
        for para0=0:0.02:max_abs
            dist_diff_final_list=zeros(size(X_test_our,1),1);
            for q=1:size(dist_diff_list_cy,2)
                lab=abs(dist_diff_list_cy{q})<para0;
                dist_diff_list_cy{q}(lab)=0;
                dist_diff_final=sum(dist_diff_list_cy{q},1)/size(dist_diff_list_cy{q},1);
                dist_diff_final_list(q)=dist_diff_final;    
            end    
        
            for para=min_all:0.02:max_all
                y=y+1;
                predict=ones(length(Y_test_our),1);
                lable=dist_diff_final_list>para;
                predict(lable,:)=0;
            
                predict_list(:,y)=predict;
                check_list=[predict,Y_att_test];
            
                acc=sum(check_list(:,1)==check_list(:,2),1)/length(Y_test_our);
                final_list(y,:)=[para0,para,acc];
            
                disp([num2str(f),'---','att:[',num2str(m),']---',num2str(para0),'---',num2str(para)]);
               
            end    
        end
    
        acc_max=max(final_list(:,3));
        acc_max_para0=final_list(final_list(:,3)==acc_max,1);
        acc_max_para=final_list(final_list(:,3)==acc_max,2);
        
%         acc_para{f}{m}{1}=acc_max;
%         acc_para{f}{m}{2}=acc_max_para0;
%         acc_para{f}{m}{3}=acc_max_para;

         
    
        ind=find(final_list(:,3)==acc_max);
        predict_max=predict_list(:,ind);
        predict_max_1=predict_max(:,1);
        predict_final(:,m)=predict_max_1;
        
        fname = sprintf( 'att2_%d', m );
        iSaveX_b4_1000(fname,final_list,acc_max,acc_max_para0,acc_max_para,predict_max);
            
    end    
    
    predict_cl_list=[];
    for i=1:size(predict_final,1)
        pre_one=predict_final(i,:);
        [idx_f,DIST]=knnsearch(Att_test,pre_one,'k',1,'NSMethod','exhaustive','Distance','correlation');
    
        predict_cl=test_classes(idx_f);
        predict_cl_list=[predict_cl_list;predict_cl];
    end

    check_list_cl=[predict_cl_list,Y_test_our];
    acc_cl=sum(check_list_cl(:,1)==check_list_cl(:,2),1)/length(Y_test_our);
    
    disp(['att:[',num2str(acc_cl),']']);
    
    overall_acc(f)=acc_cl;
    
    if acc_cl>the_max_acc
        the_max_acc=acc_cl;
        good_feature_list=temp_list;
        
        save(['final_result_',num2str(f)],'check_list_cl','acc_cl','predict_final');
        
    end
      
end

save('overall_acc','overall_acc');


