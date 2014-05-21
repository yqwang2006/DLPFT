#pragma once

namespace dlpft{
	namespace factory{
		// ���幤����  
		// ProductΪ�����Ʒ��ConcreteProductΪ�����Ʒ  
		template<typename Product, typename ConcreteProduct>  
		class ConcreteCreator  
		{  
		public:  
			// ����һ�������Ʒ  
			static Product* createProduct()  
			{  
				return new ConcreteProduct();  
			}  
		};  
	};
};
