#pragma once
#include <map>
#include "ConcreteCreator.h"
namespace dlpft{
	namespace factory{
		// ���󹤳�  
		// ProductΪ�����Ʒ  
		template<typename Product>  
		class Creator  
		{  
			// ����ʵ��  
		public:  
			static Creator& Instance()  
			{  
				static Creator<Product> instance;  
				return instance;  
			}  
		private:  
			Creator() {}  
			~Creator() {}  
			Creator(Creator&);  
			// ����ӿ�  
		public:  
			typedef Product* (*CreateProductDelegate)( ); // ������Ʒ�ĺ���ί��  
			typedef std::map<std::string, CreateProductDelegate> MapRegisterCreatorItem;  
			// ���ݾ����Ʒ���ɾ��幤��  
			// ������ע������󹤳�  
			// ConcreteProductΪ�����Ʒ  
			template<typename ConcreteProduct>  
			void registerCreator(const std::string& _type)  
			{  
				mConcreteCreators[_type] = ConcreteCreator<Product, ConcreteProduct>::createProduct;  
			}  
			// ɾ�����о��幤��  
			void unregisterAllCreators()  
			{  
				mConcreteCreators.clear();  
			}  
			// ��������Ϊ_type�Ĳ�Ʒ  
			// ʧ�ܷ���0  
			Product* createProduct(const std::string& _type)  
			{  
				MapRegisterCreatorItem::iterator type = mConcreteCreators.find(_type);  
				if (type != mConcreteCreators.end())  
				{  
					CreateProductDelegate create = type->second;  
					if (create != 0)  
						return create();  
				}  
				return 0;  
			}  
		private:  
			MapRegisterCreatorItem mConcreteCreators; // ��������ע����ľ��幤��  
		};  
	};
};