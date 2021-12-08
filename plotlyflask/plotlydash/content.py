def noidungbieudo(chart_name):
        if chart_name == "Line":          
            noidung_bieudo = '''Biểu đồ đường thẳng được cho là hình thức đơn giản nhất của biểu đồ khi nói 
            đến thị trường tài chính, được sử dụng trong quá khứ bởi các nhà giao dịch chứng khoán. 
            Chúng được dựa trên các dòng được vẽ ra từ một giá đóng cửa một phiên đến giá đóng cửa phiên tiếp theo. 
            Đây là một cách dễ dàng để thể hiện sự chuyển động giá chung của một thị trường trong một khoảng thời gian cụ thể. 
            Bởi vì sự đơn giản của chúng, biểu đồ đường cũng giúp nhận ra các xu hướng thị trường và thường được ưa thích bởi người mới. 
            Nếu  đang dự định để bắt đầu trên các thị trường tài chính, hãy thử thực hành trên biểu đồ đường thẳng.'''
    # Candelstick
        elif chart_name == "Candlestick":
            noidung_bieudo = '''
            
            Biểu đồ nến thể hiện cùng một thông tin, nhưng được cho là trực quan và dễ tiếp cận hơn.
            Những cây nến đề cập tới phạm vi cao thấp với các giá mở và đóng. Giá cao nhất được chỉ ra bởi một tim nến phía trên,
            trong khi giá thấp nhất được hiển thị bởi tim nến bên dưới. Các thân nến càng dài, áp lực của việc mua hoặc bán càng lớn.
            Điều đó có nghĩa rằng thân nến càng dài, sự biến động tỷ giá càng lớn. Ngược lại, nến ngắn cho thấy sự biến động giá rất
            ít và biểu tượng cho thị trường sắp rơi vào thế tích lũy (khoảng thời gian mà trường tương đối yên bình).
            
            Đường SMA (hay Simple Moving Average) là đường trung bình động đơn giản  được tính bằng trung bình cộng các mức
            giá đóng cửa trong một khoảng thời gian giao dịch nhất định..
            
            Cách tính đường SMA
            Đường MA dùng trong dài hạn: SMA(100); SMA(200)
            Đường MA dùng trong trung hạn: SMA(50)
            Đường MA dùng trong ngắn hạn: SMA(10), SMA(14), SMA(20)
                                                                
            Ý nghĩa của đường trung bình động đơn giản SMA: Đường SMA – Simple Moving Average chỉ xu hướng giá của cổ phiếu, do đó đường hỗ trợ chúng ta dự đoán giá trong tương lai. Nhìn vào độ dốc của đường MA bạn có thể đoán giá sẽ biến đổi như thế nào để ra quyết định tốt ưu nhất.
            
            '''
        elif chart_name == "SMA":
            noidung_bieudo ='''
            Đường trung bình MA- Moving Average gọi là đường trung bình động, được hiểu là trung bình cộng của chuỗi giá 
            trong một khoảng thời gian nhất định.

            Đường MA là phương tiện rất phổ biến trong phân tích kỹ thuật chứng khoán được nhiều người tin dùng mà bất cứ nhà phân 
            tích kỹ thuật nào cũng không nên bỏ qua. Nhờ đó, nhà đầu tư nhận biết được tín hiệu mua-bán.
        
            Đường SMA (hay Simple Moving Average) là đường trung bình động đơn giản  được tính bằng trung bình cộng 
            các mức giá đóng cửa trong một khoảng thời gian giao dịch nhất định.
            
            Các đường trung bình SMA phổ biến:

            Đường MA dùng trong dài hạn: SMA(100); SMA(200)
            Đường MA dùng trong trung hạn: SMA(50)
            Đường MA dùng trong ngắn hạn: SMA(10), SMA(14), SMA(20) 
            
            Đường SMA – Simple Moving Average chỉ xu hướng giá của cổ phiếu, do đó đường hỗ trợ chúng ta dự đoán 
            giá trong tương lai. Nhìn vào độ dốc của đường MA bạn có thể đoán giá sẽ biến đổi như thế nào để ra quyết định tốt ưu nhất.
            
            '''
        # Exponential moving average
        elif chart_name == "EMA":
            noidung_bieudo = '''
            
            Exponential Moving Average (EMA) là đường trung bình động hàm mũ được tạo ra để giải quyết vấn đề phản 
            ứng chậm với biến động của tỷ giá mà SMA (Simple Moving Average) bị hạn chế.
            
            Đường trung bình động hàm mũ (EMA) làm giảm độ trễ bằng cách chú trọng nhiều hơn vào các mức tỷ giá gần đây. 
            Trọng số được áp dụng cho các mức tỷ giá gần đây nhất phụ thuộc vào số chu kỳ (Số ngày, giờ, tuần..) được áp dụng.
            Các đường EMA khác với các đường trung bình động giản đơn vì cách tính toán EMA của một ngày phụ thuộc vào các phép 
            tính EMA cho tất cả các ngày trước ngày đó. Bạn cần nhiều hơn dữ liệu của 10 ngày để tính toán chính xác EMA 10 ngày chính xác.
            
            '''
        elif chart_name == "MACD":
            noidung_bieudo = '''  
            Đường MACD là đường được tính toán bằng chênh lệch giữa hai trung bình trượt số mũ.
            
            Nội dung
            - Thông thường, các nhà phân tích sử dụng giá trị trung bình trượt số mũ 26 ngày trừ đi giá trị trung bình trượt số mũ 12 ngày để tính toán ra đường MACD.
            
            - Giá trị MACD tính toán được dựa vào hai trung bình trượt số mũ này sẽ dao động quanh mức 0.
            
            - Một đường MACD dương có nghĩa là giá trung bình trong 12 ngày trước đó lớn hơn giá trung bình của 26 ngày trước đó.
            
            Đường MACD thường được vẽ ở phía dưới của đồ thị phân tích kĩ thuật với đường dấu hiệu. Đường dấu hiệu (signal line) là đường trung bình trượt số mũ của MACD, thường là đường trung bình trượt số mũ 9 ngày.
        
        
            Ý nghĩa của chỉ báo đường MACD
            Đường MACD tỏ ra hữu hiệu khi trên thị trường có xu hướng rõ rệt. 
            
            Đường MACD lớn hơn 0 là dấu hiêu cho thấy đường trung bình trượt ngắn hạn nằm ở phía trên đường trung bình trượt dài hạn hơn.
            
            Ngược lại, MACD nhỏ hơn 0, đường trung bình trượt ngắn hạn nằm ở phía dưới đường trung bình trượt dài hạn hơn.
                        
            '''
            
        elif chart_name == "RSI":
            noidung_bieudo = '''
            là chỉ báo động lượng đo lường mức độ thay đổi giá gần đây, nhằm đánh giá việc mua quá mức hoặc bán quá mức 
            ở một mức giá của 1 cổ phiếu hoặc các tài sản tài chính khác.
            
            Đường RSI là chỉ báo phân tích kỹ thuật được nhiều NĐT Việt Nam sử dụng, và được hiển thị dưới dạng biểu đồ giao động từ 0 đến 100
            
            Dù đường RSI chuyển động qua lại giữa 2 mức: 0 và 100. Tuy nhiên có 2 khu vực chính khi sử dụng đường RSI là: Vùng quá mua & Vùng quá bán.

            Vùng quá mua (overbought): Khi đường RSI vượt ngưỡng 70, lúc này tín hiệu đường RSI cho thấy nhà đầu tư là muốn mua quá nhiều, đẩy vượt quá xa so với ngưỡng cân bằng.
            Vùng quá bán (oversold): Khi đường RSI dưới ngưỡng 30, lúc này đường RSI cho thấy nhà đầu tư bán quá nhiều, đẩy giá quá thấp so với ngưỡng cân bằng.
                        
            Khi mức giá cổ phiếu ở vùng quá mua hoặc quá bán, thì khả năng cổ phiếu sẽ điều chỉnh để có một mức giá phù hợp và cân bằng.

            Nếu cổ phiếu đạt mức quá mua liên tục và duy trì trên 70, đó thường là cổ phiếu đang giai đoạn tăng mạnh, thì mức điều chỉnh 70 sẽ lên thành 80. Lưu ý thêm: Trong các xu hướng mạnh, chỉ báo RSI có thể ở trạng thái quá mua hoặc quá bán trong thời gian dài.

            Tuy nhiên, nhìn chung:

            Tín hiệu bán: Khi giá cổ phiếu ở vùng quá mua, và đường RSI rớt dưới ngưỡng 70, bởi đó là dấu hiệu giá cổ phiếu có khả năng giảm lớn hơn lớn hơn khả năng tăng giá.
            Tín hiệu mua: Khi giá cổ phiếu ở vùng quá bán, và đường RSI vượt qua ngưỡng 30, bởi đó là dấu hiệu giá cổ phiếu có khả năng tăng giá lớn hơn khả năng giảm giá.
            Lưu ý:

            Trong một thị trường tăng giá mạnh hoặc uptrend, đường RSI có xu hướng duy trì trong phạm vi từ 40 đến 90. Khi đó vùng 40-50 đóng vai trò là vùng hỗ trợ.

            Ngược lại, trong 1 xu hướng giảm mạnh hay downtrend, đường RSI có xu hướng ở phạm vi từ 10-60, khi đó vùng 50-60 đóng vai trò là ngưỡng kháng cự.
            '''

        else:
            noidung_bieudo = ''''''
        return noidung_bieudo 
