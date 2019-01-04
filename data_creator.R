data = read.csv('loan_stat542.csv')

all.test.id = read.csv('Project3_test_id.csv')

i = 3 #column of test id df
test_data_initial = data[data$id %in% all.test.id[,i],]
x = levels(test_data_initial$loan_status)
test_data_initial$loan_status = as.integer(as.character(mapvalues(test_data_initial$loan_status, from = x, to = c(1,1,0))))
labels = data.frame(id = test_data_initial$id, y = test_data_initial$loan_status)


#dropping Y for test
drop = c('loan_status')
test_data = test_data_initial[,!(names(test_data_initial)%in%drop)]

train_data = data[!data$id %in% all.test.id[,i],]

write.csv(test_data,'test.csv',row.names = FALSE)
write.csv(train_data,'train.csv',row.names = FALSE)
write.csv(labels, 'label.csv', row.names = FALSE)

