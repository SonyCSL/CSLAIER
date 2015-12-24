var editor; // コードエディット時のeditorオブジェクト

$(function(){
    if($('#index').text() !== "") {
        setInterval(check_train_progress, 60000);
    }
    if($('#model_detail').text() !== "" || $('#new_model').text() !== "") {
        createEditor();
    }
    if($('#model_detail').text() !== '') {
        var hash = location.hash;
        if(hash == '#result') {
            showResultScreen();
        }
    }
    if($('#gpu_meter_needed').text() !== '') {
        _.each(gpus, function(gpu){
            var iframe = document.createElement('iframe');
            iframe.width = 228;
            iframe.height = 228;
            iframe.frameBorder = 0;
            iframe.style.cssText = 'border: none';
            var url = '/statics/html/gpu_usage.html?';
            var keys = Object.keys(gpu);
            var uuid = '';
            _.each(keys, function(key){
                if(key == 'uuid') uuid = gpu[key];
                url += '&' + encodeURIComponent(key) + '=' + encodeURIComponent(gpu[key]);
            });
            url = url.replace('?&', '?');
            iframe.src = url;
            $('#' + uuid).append($(iframe));
        });
    }
});

$('#uploadDataset #submit_dataset').on('click', function(e){
    if($('#submit_dataset').hasClass('disabled')) return;
    $('#upload_modal').modal('hide');
    $('#uploading_progress_div').removeClass('hidden');
    $('body').addClass('noscroll');
    var fd = new FormData();
    fd.append('dataset_name', $('#uploadDataset #dataset_name_input').val());
    fd.append('fileInput', $('#uploadDataset #fileInput').prop('files')[0]);
    $.ajax({
        async: true,
        xhr: function(){
            XHR = $.ajaxSettings.xhr();
            if(XHR.upload){
                XHR.upload.addEventListener('progress', function(e){
                    var progress_rate = ~~(parseInt(e.loaded/e.total*10000, 10)/100) ;
                    $('#progress-bar')
                        .attr('aria-valuenow', progress_rate)
                        .css('width', progress_rate + '%')
                        .html('<span class="sr-only">' + progress_rate +'% Complete</span>');
                    $('#progress_rate').text(progress_rate + '%');
                }, false);
            }
            return XHR;
        },
        url: "/api/upload",
        type: "POST",
        data: fd,
        contentType: false,
        processData: false
    })
    .done(function(){
        location.reload();
    })
    .fail(function(jqXHR, textStatus, errorThrown){
        console.log(errorThrown);
        $('#uploading_progress_div').addClass('hidden');
        $('body').removeClass('noscroll');
        alert('Could not upload Dataset.');
    });
});

var check_train_progress = function(){
    $.get('/api/models/chekc_train_progress', function(ret){
        _.each(ret.progress, function(p){
            var target = $('#model_' + p.id);
            switch(p.is_trained) {
                case 0:
                    target.removeAttr('class').addClass('model model-nottrained');
                    target.find('.progress-info').html('<span class="label label-nottrained">Not Trained</span>');
                    break
                case 1:
                    target.removeAttr('class').addClass('model model-progress');
                    target.find('.progress-info').html('<span class="label label-progress">In Progress</span>');
                    break;
                case 2:
                    target.removeAttr('class').addClass('model');
                    target.find('.progress-info').html('<span class="label label-trained">Trained</span>');
                    break;
                default:
                    break;
            }
        });
    });
};

$('#uploading_progress_div').on('click', function(e){
    e.preventDefault();
});

$('#dataset_name_input').on('keyup', function(e){
    if(/^[\w][\w|\ |\-]*$/.test($(this).val())){
        $('#submit_dataset').removeClass('disabled');
    } else {
        $('#submit_dataset').addClass('disabled');
    }
    
});

$('.dataset').on('click', function(e){
    var dataset_id = $(this).data('id');
    location.href = '/dataset/show/' + dataset_id;
});

$('.category').on('click', function(e){
    var path = $(this).data('path');
    var dataset_id = $(this).data('id');
    if(path.indexOf('/') !== 0) path = '/' + path
    location.href = '/dataset/show/' + dataset_id + path;
});

$('.category-image').on('click', function(e){
    if(window.confirm('Is it okay to remove this image?')) {
        var form = '<input type="hidden" name="file_path" value="'+ $(this).data('path') +'">';
        $('<form action="/dataset/delete/file/' + $("#dataset-id").val() + '/' + $("#dataset-path").val() + '" method="POST">' + form + '</form>').append('body').submit();
    }
});

$('#btn_delete_category').on('click', function(e){
    if(window.confirm('Is it okay to remove this category? Images will be also removed.')) {
        var form = '<input type="hidden" name="category_path" value="'+ $(this).data('path') +'">';
        $('<form action="/dataset/delete/category/' + $("#dataset-id").val() + '" method="POST">' + form + '</form>').append('body').submit();
    }
});

$('#input_category_name').on('keyup', function(e){
    var category_name = $(this).val();
    if(/^[a-zA-A0-9][\w-_]*$/.test(category_name) && categories.indexOf(category_name) < 0) {
        $('#create_category_submit').removeClass('disabled');
    } else {
        $('#create_category_submit').addClass('disabled');
    }
});

$('#create_category_submit').on('click', function(e){
    $.post('/dataset/create/category/' + $('#dataset_id').val(), {category_name: $('#input_category_name').val()}, function(ret){
        location.reload();
    });
});

$('#btn_delete_dataset').on('click', function(e){
    if(window.confirm('Is it okay to remove this Dataset? Images will be also removed.')) {
        location.href = '/dataset/remove/' + $('#dataset_id').val();
    }
});

$('#model_template_list').on('change', function(e){
    var model_name = $(this).val();
    $.get('/api/models/get_model_template/' + model_name, function(ret){
        var now = moment().format('YYYYMMDDHHmmss');
        $('#model_name_input').val(now + model_name);
        $('#network_name_input').val(model_name);
        $('#network_edit_area').val(ret.model_template);
        createEditor();
    });
});

$('#create_model_form').submit(function(){
    if(!/^[\w-\.]+$/.test($('#model_name_input').val())) {
        alert('Use Alphabet or Numbers on Model Name.')
        return false;
    } 
    if(/^\s*$/.test($('#network_edit_area').val())){
        alert('Network definition is needed');
        return false;
    }
});

$('#create_model_reset').on('click',function(e){
    $('#model_name_input').val('');
    $('#network_name_input').val('');
    $('#algorithm_name_input').val('');
    $('#network_edit_area').val('');
    createEditor();
    $('#model_template_list').val('');
});

$('.model').on('click', function(e){
    if($(this).hasClass('model-nottrained')) {
        location.href = '/models/show/' + $(this).data('modelid');
    } else {
        location.href = '/models/show/' + $(this).data('modelid') + '#result';
    }
});

$('#epoch_select').on('keypress, change', function(e){
    var max = parseInt($(this).attr('max'), 10);
    var current_val = parseInt($(this).val(), 10);
    $('#epoch_on_modal').val(current_val);
    $('#epoch_on_modal_title').text("Epoch:" + current_val);
    if(0 < current_val && current_val <= max) {
        $(this).parent().removeClass('has-error');
        $(this).parent().addClass('has-feedback has-success');
        $('button.need-epoch').removeClass("disabled");
    } else {
        $(this).parent().removeClass('has-success');
        $(this).parent().addClass('has-feedback has-error');
        $('button.need-epoch').addClass("disabled");
    }
});

$('#start_train_btn').on('click', function(e){
    $('#start_train_modal').modal('hide');
    $('#processing_screen').removeClass('hidden');
    var model_id = $('#model_id').val();
    var dataset_id = parseInt($('#select_dataset').val(), 10);
    var epoch = $('#epoch_input').val();
    var gpu_num = $('#gpu_num').val();
    if(dataset_id < 0) {
        alert('Select Dataset.');
        return;
    }
    $.post('/models/start/train',{
            model_id: model_id,
            dataset_id: dataset_id,
            epoch: epoch,
            gpu_num: gpu_num
        }, function(ret){
        if(ret.status === "OK") {
            $('#processing_screen').addClass('hidden');
            $('#start_train_div').addClass('hidden');
            $('#model_detail_buttons').addClass('hidden');
            $('span.label.label-nottrained')
                .removeClass('label-nottrained')
                .addClass('label-progress')
                .text('In Progress');
            $('span.label.label-trained')
                .removeClass('label-trained')
                .addClass('label-progress')
                .text('In Progress');
            showResultScreen();
            location.hash = "result";
            return;
        }
        console.log(ret.traceback);
        alert('Failed to start train.');
        $('#processing_screen').addClass('hidden');
        return;
    })
    .fail(function(){
        alert('Failed to start train.');
        $('#processing_screen').addClas('hidden');
    });
});

$('#delete_model_button').on('click', function(e){
    if(window.confirm('Is it okay to remove this model?')) {
        var model_id = $('#model_id').val();
        var form = '<input type="hidden" name="model_id" value="'+ model_id +'">';
        $('<form action="/models/delete/' + model_id + '" method="POST">' + form + '</form>').append('body').submit();
    }
});

$('#processing_screen').on('click', function(e){
    e.stopPropagation();
});

$('#model_dl_btn').on('click', function(e){
    var model_id = $('#model_id').val();
    var epoch = $('#epoch_select').val();
    window.open('/models/download/' + model_id + '/' + epoch);
});

$('#mean_dl_btn').on('click', function(e){
    var model_id = $('#model_id').val();
    window.open('/models/mean/download/' + model_id);
});

$('#label_dl_btn').on('click', function(e){
    var model_id = $('#model_id').val();
    window.open('/models/labels/download/' + model_id);
});

$('#graph_tab').on('click', function(e){
    $(this).addClass('active');
    $('#model_detail_graph').removeClass('hidden');
    $('#network_tab').removeClass('active');
    $('#model_detail_network').addClass('hidden');
});

$('#network_tab').on('click', function(e){
    $(this).addClass('active');
    $('#model_detail_network').removeClass('hidden');
    $('#model_detail_graph').addClass('hidden');
    $('#graph_tab').removeClass('active');
});


$('#model_edit_cancel').on('click', function(e){
    $('#network_edit_area').text($('#original_network').text());
});

$('#create_new_network_modal').on('show.bs.modal', function(e){
    var original_name = $('#original_name').val();
    original_name = original_name.replace('.py', '');
    var temp_name_arr = original_name.split('_');
    if(/^\d+$/.test(temp_name_arr[temp_name_arr.length - 1])){
        var temp_index = parseInt(temp_name_arr.pop(), 10) + 1;
        temp_name_arr.push(temp_index);
    } else {
        temp_name_arr.push('1');
    }
    var name = temp_name_arr.join('_');
    $('#modal_create_network_name').val(name + '.py');
});

$('#network_edit_area').on('keydown', function(e){
    $('#create_network_buttons').removeClass('hidden');
});

$('#create_new_network_modal').on('show.bs.modal', function(e){
    $('#modal_create_network_my_network').val($('#network_edit_area').val());
});

$('#create_new_network_modal_form').submit(function(){
    if(!/^[\w-\.]+$/.test($('#modal_create_network_name').val())) {
        alert('Use Alphabet or Numbers on Model Name.')
        return false;
    } 
    if(/^\s*$/.test($('#modal_create_network_my_network').val())){
        alert('Network definition is needed');
        return false;
    }
});

$('#graph_tab').on('click', function(e){
    draw_train_graph();
    setInterval("draw_train_graph()", 30000);
});


var draw_train_graph = function(){
    var model_id = $('#model_id').val();
    $.get('/api/models/get_training_data/' + model_id, function(ret){
        if(ret.status != 'ready') return;
        $('#training_graph').empty();
        // スケールと出力レンジの定義
        var margin = {top: 20, right: 20, bottom: 30, left: 50};
        var width = 550 - margin.left - margin.right;
        var height = 450 - margin.top - margin.bottom;
        
        var xEpoch       = d3.scale.linear().range([0, width]);
        var xCount       = d3.scale.linear().range([0, width]);    
        var yLoss        = d3.scale.linear().range([height, 0]);
        var yValLoss     = d3.scale.linear().range([height, 0]);
        var yAccuracy    = d3.scale.linear().range([height, 0]);
        var yValAccuracy = d3.scale.linear().range([height, 0]);
        
    
        // 軸の定義
        var xAxis         = d3.svg.axis()
                            .scale(xEpoch)
                            .orient("bottom")
                            .innerTickSize(-height)
                            .outerTickSize(0)
                            .tickPadding(10);
        var yAxisLoss     = d3.svg.axis().scale(yLoss).orient("left");
        var yAxisAccuracy = d3.svg.axis().scale(yAccuracy).orient("right");
    
        // 線の定義
        var lineLoss = d3.svg.line()
                .x(function(d) { return xCount(d.count); })
                .y(function(d) { return yLoss(d.loss); });
        var lineValLoss = d3.svg.line()
                .x(function(d) { return xCount(d.count)})
                .y(function(d) { return yValLoss(d.val_loss)});
        var lineAccuracy = d3.svg.line()
                .x(function(d) { return xCount(d.count); })
                .y(function(d) { return yAccuracy(d.accuracy); });
        var lineValAccuracy = d3.svg.line()
                .x(function(d) { return xCount(d.count); })
                .y(function(d) { return yValAccuracy(d.val_accuracy); });
                
        var svg = d3.select("#training_graph").append('svg')
            .attr('width', 630).attr('height', 460)
            .append('g').attr("transform", "translate(0,0)");;
        var parsedData = d3.tsv.parse(ret.data, function(){
            var count = -1;
            return function(data){
                count++;
                var SAMPLING_RATE = 10;
                if(!data['loss(val)'] && count % SAMPLING_RATE != 0) return;
                data.count = count;
                data.epoch = +data.epoch;
                data.loss = data.loss ? +data.loss : null;
                data.accuracy = data.accuracy ? +data.accuracy : null;
                data.val_loss = data['loss(val)'] ? +data['loss(val)'] : null;
                data.val_accuracy = data['accuracy(val)'] ? +data['accuracy(val)'] : null;
                return data;
            };
        }());
        
        var train_accuracy_data = _.filter(parsedData, function(obj){
            if(obj.accuracy) return true;
        });
        var train_loss_data = _.filter(parsedData, function(obj){
            if(obj.loss) return true;
        });
        var val_accuracy_data = _.filter(parsedData, function(obj){
            if(obj.val_accuracy) return true;
        });
        var val_loss_data = _.filter(parsedData, function(obj){
            if(obj.val_loss) return true;
        });
        
        xEpoch.domain(d3.extent(parsedData, function(d) { return d.epoch; }));
        xCount.domain(d3.extent(parsedData, function(d) { return d.count}));
        yLoss.domain(d3.extent(train_loss_data, function(d) { return d.loss; }));
        yAccuracy.domain(d3.extent(train_accuracy_data, function(d) { return d.accuracy; }));
        yValLoss.domain(d3.extent(val_loss_data, function(d) { return d.val_loss; }));
        yValAccuracy.domain(d3.extent(val_accuracy_data, function(d) { return d.val_accuracy; }));
        
        // loss
        svg.append("path")
            .datum(train_loss_data)
            .attr("class", "line-loss")
            .attr("transform", "translate(" + margin.left + ",0)")
            .attr("d", lineLoss);
        // accuracy
        svg.append("path")
            .datum(train_accuracy_data)
            .attr("class", "line-accuracy")
            .attr("transform", "translate(" + margin.left + ",0)")
            .attr("d", lineAccuracy);
        // loss(val)
        svg.append("path")
            .datum(val_loss_data)
            .attr("class", "line-val-loss")
            .attr("transform", "translate(" + margin.left + ",0)")
            .attr("d", lineValLoss);
        // accuracy(val)
        svg.append("path")
            .datum(val_accuracy_data)
            .attr("class", "line-val-accuracy")
            .attr("transform", "translate(" + margin.left + ",0)")
            .attr("d", lineValAccuracy);
        // x axis(epoch)
        svg.append("g")
            .attr("class", "x axis")
            .attr("fill", "white")
            .attr("transform", "translate(" + margin.left + ",410)")
            .call(xAxis);
        // y axis left side(loss)
        svg.append("g")
            .attr("class", "y axis")
            .attr("transform", "translate(" + margin.left + ",0)")
            .attr("fill", "white")
            .call(yAxisLoss)
            .append("text")
            .attr("y", 6)
            .attr("x", 5)
            .attr("dy", ".71em")
            .attr("fill", "white")
            .style("text-anchor", "start")
            .text("loss");
        // y axis right side(accuracy)
        svg.append("g")
            .attr("class", "y axis")
            .attr("transform", "translate(" + (width + margin.left + 10)+",0)")
            .attr("fill", "white")
            .call(yAxisAccuracy)
            .append("text")
            .attr("y", 6)
            .attr("x", -5)
            .attr("dy", ".71em")
            .attr("fill", "white")
            .style("text-anchor", "end")
            .text("accuracy")
        // legend
        var legend_data = [
            {title: "loss", color:"steelblue"},
            {title: "accuracy", color:"orange"},
            {title: "loss(val)", color:"#0c0"},
            {title: "accuracy(val)", color:"red"}
        ];
        _.each(legend_data, function(d, i){
            addLegend(svg, d.title, d.color, i);
        });
    });
};

var addLegend = function(svg, title, color, i){
    var legend = svg.append('g')
        .attr('class', 'legend')
        .attr('transform', 'translate(0,' + (14*i+5) + ')');
    legend.append('rect')
        .attr('width', 10)
        .attr('height', 10)
        .style('fill', color);
    legend.append('text')
        .attr('x', 14)
        .attr('y', 10)
        .attr("font-size", ".8em")
        .attr('fill', color)
        .style('text-anchor', 'start')
        .text(title);
};

var createEditor = function(){
    $('.CodeMirror').remove();
    if($('#network_edit_area').prop('tagName') == 'TEXTAREA') {
        editor = CodeMirror.fromTextArea(document.getElementById('network_edit_area'),{
            mode: "python",
            lineNumbers: true,
            indentUnit: 4
        });
    } else {
        var code = $('#network_edit_area').text();
        $('#network_edit_area').empty();
        editor = CodeMirror(document.getElementById('network_edit_area'), {
            mode: "python",
            lineNumbers: true,
            indentUnit: 4,
            readOnly: true,
            value: code
        });
    }

    editor.on("change", function(){
        editor.save();
        $('#create_network_buttons').removeClass('hidden');
    });
    if(editor.save) editor.save();
};

var showResultScreen = function(){
        $('#network_tab').removeClass('active');
        $('#graph_tab').addClass('active');
        $('#model_detail_network').addClass('hidden');
        $('#model_detail_graph').removeClass('hidden');
        draw_train_graph();
        setInterval("draw_train_graph()", 30000);
};

